//===- SoftPointerAuth.cpp - Software lowering for ptrauth intrinsics -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers the llvm.ptrauth intrinsics into something that can
// be supported (inefficiently) on an arbitrary target.
//
// The runtime functions you must define to use this pass are:
//   /// Apply a signature to the given unsigned pointer value.
//   void *__ptrauth_sign(void *pointer, int32_t key, uintptr_t discriminator);
//
//   /// Remove the signature from the given signed pointer value.
//   void *__ptrauth_strip(void *pointer, int32_t key);
//
//   /// Authenticate and remove the signature on the given signed
//   /// pointer value.  Trap on authenticate failure.
//   void *__ptrauth_auth(void *pointer, int32_t key, uintptr_t discriminator);
//
//   /// Blend a small non-zero value into a primary discriminator,
//   /// which is expected to resemble a pointer.
//   uintptr_t __ptrauth_blend(uintptr_t primaryDiscriminator,
//                             uintptr_t secondaryDiscriminator);
//
//   /// Compute a full, pointer-wide signature on a value.
//   uintptr_t __ptrauth_sign_generic(uintptr_t data, uintptr_t discriminator);
//
// The resulting code pattern does not perfectly protect against the backend
// inserting code between authentications and uses, and so the result may
// be attackable.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/SoftPointerAuth.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <map>

#define DEBUG_TYPE "soft-ptrauth"

using namespace llvm;
using IRBuilderTy = llvm::IRBuilder<>;

namespace {

/// A structure for tracking uses of relocations within a Constant.
struct UseSite {
  /// A map from operand index to the tracking sites of children.
  /// If this is empty, the Constant is a GlobalVariable for a relocation.
  /// Otherwise, the Constant is a ConstantAggregate or ConstantExpr, and
  /// the relocation reference(s) appear further down in the tree.
  std::map<unsigned, UseSite> Children;
};

/// A linked list down to the use of a relocation.
struct UsePath {
  const UsePath *Next;
  unsigned OperandIndex;
};

enum TypeTag {
  IntPtr,                  // uintptr_t
  Discriminator = IntPtr,  // uintptr_t
  Key,                     // uint32_t
  VoidPtr,                 // i8*
};

class SoftPointerAuth {
  // The module.
  Module *M = nullptr;

  // Cached function pointers, initialized lazily.
  FunctionCallee SignPointerFn = nullptr;
  FunctionCallee AuthPointerFn = nullptr;
  FunctionCallee StripPointerFn = nullptr;
  FunctionCallee BlendDiscriminatorFn = nullptr;
  FunctionCallee SignGenericFn = nullptr;

  Optional<IRBuilderTy> GlobalConstructorBuilder;

public:
  SoftPointerAuth() {}

  bool runOnModule(Module &M);

private:
  bool isPointerAuthRelocation(GlobalVariable *global);

  bool transformRelocations();
  void transformGlobalInitializer(GlobalVariable *global,
                                  const UseSite &usesToTransform);
  Constant *transformInitializer(GlobalVariable *global,
                                 SmallVectorImpl<Constant*> &pathToInitializer,
                                 Constant *initializer,
                                 const UseSite &usesToTransform);
  void transformInstructionOperands(Instruction *user,
                                    const UseSite &usesToTransform);
  Value *emitTransformedConstant(IRBuilderTy &builder, Constant *constant,
                                 const UseSite &usesToTransform);
  IRBuilderTy &continueGlobalConstructor();

  bool transformCalls();
  bool transformCall(CallInst *call);
  bool transformInvoke(InvokeInst *call);
  bool transformPointerAuthCall(CallBase *oldCall,
                                const OperandBundleUse &bundle);

  Value *emitSign(IRBuilderTy &builder, Value *pointer,
                  Value *key, Value *discriminator);
  Value *emitResign(IRBuilderTy &builder, Value *pointer,
                    Value *oldKey, Value *oldDiscriminator,
                    Value *newKey, Value *newDiscriminator);
  Value *emitAuth(IRBuilderTy &builder, Value *pointer,
                  Value *key, Value *discriminator);
  Value *emitStrip(IRBuilderTy &builder, Value *pointer, Value *key);
  Value *emitBlend(IRBuilderTy &builder, Value *primary, Value *secondary);
  Value *emitSignGeneric(IRBuilderTy &builder,
                         Value *value, Value *discriminator);

  /// Check whether the callee of a call has the right prototype.
  bool hasExpectedPrototype(CallBase *call, TypeTag resultTypeTag,
                            ArrayRef<TypeTag> argTypeTags) {
    if (!hasType(call, resultTypeTag))
      return false;

    if (call->arg_size() != argTypeTags.size())
      return false;
    for (unsigned i = 0, e = argTypeTags.size(); i != e; ++i) {
      if (!hasType(call->getArgOperand(i), argTypeTags[i]))
        return false;
    }
    return true;
  }

  /// Does the given value have its expected type?
  bool hasType(Value *value, TypeTag tag) {
    auto type = value->getType();
    switch (tag) {
    case VoidPtr:
      if (auto ptrType = dyn_cast<PointerType>(type))
        return ptrType->getAddressSpace() == 0 &&
               ptrType->getElementType()->isIntegerTy(8);
      return false;
    case Key:
      return type->isIntegerTy(32);
    case IntPtr:
      return type->isIntegerTy(M->getDataLayout().getPointerSizeInBits(0));
    }
    llvm_unreachable("unexpected type tag");
  }
  /// Fetch an expected type.
  Type *getType(TypeTag tag) {
    switch (tag) {
    case VoidPtr: return Type::getInt8PtrTy(M->getContext());
    case Key: return Type::getInt32Ty(M->getContext());
    case IntPtr: return Type::getIntNTy(M->getContext(),
                                  M->getDataLayout().getPointerSizeInBits(0));
    }
    llvm_unreachable("unexpected type tag");
  }

  ConstantInt *getInt32(unsigned value) {
    return ConstantInt::get(Type::getInt32Ty(M->getContext()), value);
  }

  /// Create a declaration for the given runtime function.
  FunctionCallee getOrInsertFunction(StringRef name, TypeTag resultTypeTag,
                                     ArrayRef<TypeTag> argTypeTags) {
    auto resultType = getType(resultTypeTag);
    SmallVector<Type*, 4> argTypes;
    for (auto argTypeTag : argTypeTags)
      argTypes.push_back(getType(argTypeTag));
    auto functionType = FunctionType::get(resultType, argTypes, false);
    return M->getOrInsertFunction(name, functionType);
  }

  FunctionCallee getSignPointerFn() {
    if (!SignPointerFn)
      SignPointerFn = getOrInsertFunction("__ptrauth_sign", VoidPtr,
                                          { VoidPtr, Key, Discriminator });
    return SignPointerFn;
  }

  FunctionCallee getAuthPointerFn() {
    if (!AuthPointerFn)
      AuthPointerFn = getOrInsertFunction("__ptrauth_auth", VoidPtr,
                                          { VoidPtr, Key, Discriminator });
    return AuthPointerFn;
  }

  FunctionCallee getStripPointerFn() {
    if (!StripPointerFn)
      StripPointerFn = getOrInsertFunction("__ptrauth_strip", VoidPtr,
                                           { VoidPtr, Key });
    return StripPointerFn;
  }

  FunctionCallee getBlendDiscriminatorFn() {
    if (!BlendDiscriminatorFn)
      BlendDiscriminatorFn = getOrInsertFunction("__ptrauth_blend",
                                                 Discriminator,
                                          { Discriminator, Discriminator });
    return BlendDiscriminatorFn;
  }

  FunctionCallee getSignGenericFn() {
    if (!SignGenericFn)
      SignGenericFn = getOrInsertFunction("__ptrauth_sign_generic", IntPtr,
                                          { IntPtr, Key, Discriminator });
    return SignGenericFn;
  }
};

} // end anonymous namespace

bool SoftPointerAuth::runOnModule(Module &M) {
  assert(!GlobalConstructorBuilder);

  // Reset any existing caches.
  SignPointerFn = nullptr;
  AuthPointerFn = nullptr;
  StripPointerFn = nullptr;
  BlendDiscriminatorFn = nullptr;
  SignGenericFn = nullptr;

  this->M = &M;

  bool changed = false;

  // Transform all of the intrinsic calls and operand bundles.
  // Doing this before transforming the relocations doesn't deeply matter,
  // but this pass has to walk all the functions and the relocation pass is
  // based on use lists, so this order minimizes redundant work.
  changed |= transformCalls();

  // Next, transform all the uses of relocations.
  changed |= transformRelocations();

  return changed;
}

/*****************************************************************************/
/********************************** Common ***********************************/
/*****************************************************************************/

Value *SoftPointerAuth::emitSign(IRBuilderTy &builder, Value *pointer,
                                 Value *key, Value *discriminator) {
  auto call = builder.CreateCall(getSignPointerFn(),
                                 { pointer, key, discriminator });
  call->setDoesNotThrow();
  return call;
}

Value *SoftPointerAuth::emitResign(IRBuilderTy &builder, Value *pointer,
                                   Value *oldKey, Value *oldDiscriminator,
                                   Value *newKey, Value *newDiscriminator) {
  // This is not an unattackable code pattern, but we don't emit one for
  // call operand bundles, either.
  auto rawValue = emitAuth(builder, pointer, oldKey, oldDiscriminator);
  return emitSign(builder, rawValue, newKey, newDiscriminator);
}

Value *SoftPointerAuth::emitAuth(IRBuilderTy &builder, Value *pointer,
                                 Value *key, Value *discriminator) {
  auto call = builder.CreateCall(getAuthPointerFn(),
                                 { pointer, key, discriminator });
  call->setDoesNotThrow();
  return call;
}

Value *SoftPointerAuth::emitStrip(IRBuilderTy &builder, Value *pointer,
                                  Value *key) {
  auto call = builder.CreateCall(getStripPointerFn(),
                                 { pointer, key });
  call->setDoesNotThrow();
  return call;
}

Value *SoftPointerAuth::emitBlend(IRBuilderTy &builder, Value *primary,
                                  Value *secondary) {
  auto call = builder.CreateCall(getBlendDiscriminatorFn(),
                                 { primary, secondary });
  call->setDoesNotThrow();
  return call;
}

Value *SoftPointerAuth::emitSignGeneric(IRBuilderTy &builder, Value *value,
                                        Value *discriminator) {
  auto call = builder.CreateCall(getSignGenericFn(),
                                 { value, discriminator });
  call->setDoesNotThrow();
  return call;
}

bool SoftPointerAuth::isPointerAuthRelocation(GlobalVariable *global) {
  // After checking the name, validate the type.
  if (global->getSection() == "llvm.ptrauth") {
    if (auto init = dyn_cast_or_null<ConstantStruct>(
                                                  global->getInitializer())) {
      return (init->getNumOperands() == 4 &&
              hasType(init->getOperand(0), VoidPtr) &&
              hasType(init->getOperand(1), Key) &&
              hasType(init->getOperand(2), Discriminator) &&
              hasType(init->getOperand(3), Discriminator));
    }
  }

  return false;
}

/*****************************************************************************/
/******************************** Relocations ********************************/
/*****************************************************************************/

/// Find all the top-level uses of a constant (i.e. the uses that are not
/// ConstantAggregates or ConstantExprs) and call the given callback
/// function on them.
template <class Fn>
static void findTopLevelUsesOfConstant(Constant *constant, const UsePath *path,
                                       const Fn &callback) {
  for (auto i = constant->use_begin(), e = constant->use_end(); i != e; ++i) {
    UsePath userPath = { path, i->getOperandNo() };
    auto user = i->getUser();

    // If the user is a global variable, there's only one use we care about.
    if (isa<GlobalVariable>(user)) {
      assert(userPath.OperandIndex == 0 && "non-zero use index on global var");
      callback(user, path);

    // If the user is an instruction, remember the operand index.
    } else if (isa<Instruction>(user)) {
      callback(user, &userPath);

    // If the user is some other kind of context, recurse.
    } else if (auto userConstant = dyn_cast<Constant>(user)) {
      findTopLevelUsesOfConstant(userConstant, &userPath, callback);
    }

    // TODO: metadata uses?
  }
}

bool SoftPointerAuth::transformRelocations() {
  SmallVector<GlobalVariable *, 16> relocations;
  SmallVector<User*, 16> rootUsers;
  DenseMap<User*, UseSite> useSites;

  // Walk all the globals looking for relocations.
  for (auto &global : M->globals()) {
    if (!isPointerAuthRelocation(&global))
      continue;

    // Remember this relocation.
    relocations.push_back(&global);

    // Remember all the top-level uses of the relocation, together with
    // paths down to the use.
    findTopLevelUsesOfConstant(&global, nullptr,
        [&](User *user, const UsePath *path) {
      // Look up an entry in the users map, adding one if necessary.
      // We remember the order in which we encountered things to avoid
      // non-deterministically walking over a DenseMap.  This still leaves
      // us vulnerable to use-list ordering, but that's harder to avoid.
      auto result = useSites.try_emplace(user);
      if (result.second) rootUsers.push_back(user);

      // Fill out the path down to the use.
      UseSite *site = &result.first->second;
      for (; path; path = path->Next) {
        site = &site->Children[path->OperandIndex];
      }
      (void) site;
    });
  }

  // Bail out if we didn't find any uses.
  if (relocations.empty())
    return false;

  // Rewrite all the root users.
  for (auto user : rootUsers) {
    const auto &uses = useSites.find(user)->second;
    if (auto global = dyn_cast<GlobalVariable>(user)) {
      transformGlobalInitializer(global, uses);
    } else {
      transformInstructionOperands(cast<Instruction>(user), uses);
    }
  }

  // Destroy all the relocations.
  for (auto reloc : relocations) {
    reloc->replaceAllUsesWith(ConstantPointerNull::get(reloc->getType()));
    reloc->eraseFromParent();
  }

  // Finish the global initialization function if we started one.
  if (GlobalConstructorBuilder) {
    GlobalConstructorBuilder->CreateRetVoid();
    GlobalConstructorBuilder.reset();
  }

  return true;
}

/// Transform a global initializer that contains signing relocations.
void SoftPointerAuth::transformGlobalInitializer(GlobalVariable *global,
                                           const UseSite &usesToTransform) {
  auto oldInitializer = global->getInitializer();
  assert(oldInitializer && "global has no initializer?");

  // transformInitializer wants the indices of a GEP to the initializer
  // that it's transforming.  Seed that with a '0' to enter the global.
  SmallVector<Constant*, 4> pathToInitializer;
  pathToInitializer.push_back(getInt32(0));

  auto newInitializer = transformInitializer(global, pathToInitializer,
                                             oldInitializer, usesToTransform);

  assert(newInitializer != oldInitializer && "no changes?");
  assert(pathToInitializer.size() == 1 && "didn't balance push/pop");

  global->setInitializer(newInitializer);

  // Make the global mutable; our constant initializer will change it.
  global->setConstant(false);
}

/// Transform part of a global initializer that contains signing relocations.
Constant *SoftPointerAuth::transformInitializer(GlobalVariable *global,
                                SmallVectorImpl<Constant*> &pathToInitializer,
                                Constant *initializer,
                                const UseSite &usesToTransform) {
  auto aggregate = dyn_cast<ConstantAggregate>(initializer);

  // If the initializer is a simple reference to a relocation, or an
  // expression in terms of same, compute it in the global construction.
  if (!aggregate) {
    auto &builder = continueGlobalConstructor();

    // Compute the value.
    auto transformedInitializer =
      emitTransformedConstant(builder, initializer, usesToTransform);

    // Drill down to the current position.
    Constant *addr = global;
    if (pathToInitializer.size() != 1)
      addr = ConstantExpr::getInBoundsGetElementPtr(global->getValueType(),
                                                    addr, pathToInitializer);

    // Store the transformed vlaue to this position.
    builder.CreateStore(transformedInitializer, addr);

    // Use a null value for the global position.
    return Constant::getNullValue(initializer->getType());
  }

  // Otherwise, the initializer is a constant aggregate.  Recurse into it
  // at the appropriate positions.  The goal here is to avoid emitting the
  // entire aggregate with stores.
  assert(!usesToTransform.Children.empty()
         && "walking into wrong initializer?");

  // Copy the original elements.
  SmallVector<Constant*, 16> elts;
  elts.reserve(aggregate->getNumOperands());
  for (auto &op : aggregate->operands())
    elts.push_back(cast<Constant>(&*op));

  // Modify just the elements that we decided to modify.
  for (const auto &eltIndexAndUses : usesToTransform.Children) {
    auto eltIndex = eltIndexAndUses.first;

    // Add an index to the GEP down to this position.
    pathToInitializer.push_back(getInt32(eltIndex));

    // Rewrite the element.
    elts[eltIndex] = transformInitializer(global, pathToInitializer,
                                     elts[eltIndex], eltIndexAndUses.second);

    // Pop the previously pushed path element.
    pathToInitializer.pop_back();
  }

  // Rebuild the aggregate.
  auto type = aggregate->getType();
  if (auto structType = dyn_cast<StructType>(type)) {
    return ConstantStruct::get(structType, elts);
  } else if (auto arrayType = dyn_cast<ArrayType>(type)) {
    return ConstantArray::get(arrayType, elts);
  } else {
    return ConstantVector::get(elts);
  }
}

/// Continue emitting the global constructor function.
IRBuilderTy &SoftPointerAuth::continueGlobalConstructor() {
  // Create the global initialization function if we haven't yet.
  if (!GlobalConstructorBuilder) {
    auto &context = M->getContext();

    // Create the function.
    auto fnType = FunctionType::get(Type::getVoidTy(context),
                                    {}, false);
    Function *fn = Function::Create(fnType, Function::PrivateLinkage,
                                    "ptrauth_soft_init", M);

    // Add the function to the global initializers list.
    appendToGlobalCtors(*M, fn, 0);

    auto entryBB = BasicBlock::Create(context, "", fn);

    GlobalConstructorBuilder.emplace(entryBB);
  }
  return *GlobalConstructorBuilder;
}

void SoftPointerAuth::transformInstructionOperands(Instruction *user,
                                              const UseSite &usesToTransform) {
  assert(!usesToTransform.Children.empty()
         && "no uses to transform for instruction");

  // Handle PHIs differently because we have to insert code into the
  // right predecessor(s).
  if (auto phi = dyn_cast<PHINode>(user)) {
    for (auto &useEntry : usesToTransform.Children) {
      auto operandIndex = useEntry.first;
      auto operand = cast<Constant>(phi->getOperand(operandIndex));

      // Figure out the block this edge corresponds to.
      auto incomingValueIndex =
        PHINode::getIncomingValueNumForOperand(operandIndex);
      auto incomingBlock = phi->getIncomingBlock(incomingValueIndex);

      // Split the edge if necessary & possible.
      // Note that we don't want to change anything structurally about 'phi'.
      auto newBlock = SplitCriticalEdge(incomingBlock, phi->getParent(),
                                        CriticalEdgeSplittingOptions()
                                          .setKeepOneInputPHIs());

      // Start inserting before the terminator in the new block.
      // If a critical edge was unsplittable, this will insert the code
      // unconditionally in the origin block, which is unfortunate but
      // acceptable because sign operations cannot fail.
      auto blockToInsertInto = newBlock ? newBlock : incomingBlock;
      IRBuilderTy builder(blockToInsertInto->getTerminator());

      // Transform the value.
      auto transformedOperand =
        emitTransformedConstant(builder, operand, useEntry.second);

      // Replace the incoming value.
      phi->setIncomingValue(incomingValueIndex, transformedOperand);
    }

    return;
  }

  // Otherwise, emit immediately before the user.
  IRBuilderTy builder(user);
  for (auto &useEntry : usesToTransform.Children) {
    auto operandIndex = useEntry.first;
    auto operand = cast<Constant>(user->getOperand(operandIndex));

    auto transformedOperand =
      emitTransformedConstant(builder, operand, useEntry.second);

    // Replace the incoming value.
    user->setOperand(operandIndex, transformedOperand);
  }
}


Value *SoftPointerAuth::emitTransformedConstant(IRBuilderTy &builder,
                                                Constant *constant,
                                              const UseSite &usesToTransform) {
  // If it's a direct reference to the relocation, we're done.
  if (auto global = dyn_cast<GlobalVariable>(constant)) {
    assert(isPointerAuthRelocation(global));
    assert(usesToTransform.Children.empty() &&
           "child uses of direct relocation reference?");

    // Decompose the relocation.
    ConstantStruct *init = cast<ConstantStruct>(global->getInitializer());
    auto pointer = init->getOperand(0);
    auto key = init->getOperand(1);
    auto primaryDiscriminator = init->getOperand(2);
    auto secondaryDiscriminator = init->getOperand(3);

    // Compute the discriminator.
    Value *discriminator;
    if (primaryDiscriminator->isNullValue()) {
      discriminator = secondaryDiscriminator;
    } else if (secondaryDiscriminator->isNullValue()) {
      discriminator = primaryDiscriminator;
    } else {
      discriminator = emitBlend(builder, primaryDiscriminator,
                                secondaryDiscriminator);
    }

    // Emit a sign operation.
    auto signedValue = emitSign(builder, pointer, key, discriminator);

    // Cast back to the signed pointer type.
    return builder.CreateBitCast(signedValue, global->getType());
  }

  // If it's a constant expression, make it an instruction and rebuild
  // its operands.
  if (auto expr = dyn_cast<ConstantExpr>(constant)) {
    assert(!usesToTransform.Children.empty() &&
           "direct use of constant expression?");

    auto instruction = expr->getAsInstruction();

    for (const auto &operandIndexAndUses : usesToTransform.Children) {
      auto operandIndex = operandIndexAndUses.first;

      auto newOperand =
        emitTransformedConstant(builder, expr->getOperand(operandIndex),
                                operandIndexAndUses.second);
      instruction->setOperand(operandIndex, newOperand);
    }

    builder.Insert(instruction);
    return instruction;
  }

  // Otherwise, it should be a constant aggregate.
  // Recursively emit the transformed elements.
  auto aggregate = cast<ConstantAggregate>(constant);
  assert(!usesToTransform.Children.empty() &&
         "direct use of whole constant aggregate?");

  SmallVector<Value*, 16> elts(aggregate->op_begin(), aggregate->op_end());

  // Transform all of the children we're supposed to transform.
  for (const auto &childUseEntry : usesToTransform.Children) {
    auto &elt = elts[childUseEntry.first];
    elt = emitTransformedConstant(builder, cast<Constant>(elt),
                                  childUseEntry.second);
  }

  // Build up the aggregate value using insertelement / insertvalue
  // as appropriate.
  auto type = aggregate->getType();
  bool isVector = isa<VectorType>(type);
  Value *transformedAggregate = UndefValue::get(type);
  for (unsigned i = 0, e = aggregate->getNumOperands(); i != e; ++i) {
    if (isVector)
      transformedAggregate =
        builder.CreateInsertElement(transformedAggregate, elts[i], i);
    else
      transformedAggregate =
        builder.CreateInsertValue(transformedAggregate, elts[i], i);
  }
  return transformedAggregate;
}

/*****************************************************************************/
/*********************** Intrinsics and Operand Bundles **********************/
/*****************************************************************************/

bool SoftPointerAuth::transformCalls() {
  bool changed = false;

  for (auto fi = M->begin(), fe = M->end(); fi != fe; ) {
    auto fn = &*fi;
    ++fi;

    // Soft return authentication is technically possible (even without backend
    // support) but not currently necessary.
    if (fn->hasFnAttribute("ptrauth-returns"))
      report_fatal_error("Soft. lowering of return address auth unsupported");

    for (auto bi = fn->begin(), be = fn->end(); bi != be; ) {
      auto bb = &*bi;
      ++bi;

      for (auto ii = bb->begin(), ie = bb->end(); ii != ie; ) {
        auto instruction = &*ii;
        ++ii;

        if (auto call = dyn_cast<CallInst>(instruction)) {
          changed |= transformCall(call);
        } else if (auto invoke = dyn_cast<InvokeInst>(instruction)) {
          changed |= transformInvoke(invoke);
        }
      }
    }
  }

  return changed;
}

bool SoftPointerAuth::transformCall(CallInst *call) {
  // Handle calls with the llvm.ptrauth operand bundle attached.
  if (auto bundle = call->getOperandBundle(LLVMContext::OB_ptrauth)) {
    return transformPointerAuthCall(call, *bundle);
  }

  // Otherwise, look for our intrinsics.
  auto callee = call->getCalledFunction();
  if (!callee) return false;
  auto intrinsicInst = dyn_cast<IntrinsicInst>(call);
  if (!intrinsicInst)
    return false;
  auto intrinsic = intrinsicInst->getIntrinsicID();
  auto rebuild = [&](function_ref<llvm::Value*(IRBuilderTy&)> fn) {
    IRBuilderTy builder(call);
    auto result = fn(builder);
    call->replaceAllUsesWith(result);
    call->eraseFromParent();
    return true;
  };

  switch (intrinsic) {
  case Intrinsic::ptrauth_sign:
    if (!hasExpectedPrototype(call, VoidPtr, {VoidPtr, Key, Discriminator}))
      return false;
    return rebuild([&](IRBuilderTy &builder) {
      return emitSign(builder, call->getArgOperand(0),
                      call->getArgOperand(1), call->getArgOperand(2));
    });

  case Intrinsic::ptrauth_resign:
    if (!hasExpectedPrototype(call, VoidPtr, {VoidPtr, Key, Discriminator,
                                              Key, Discriminator}))
      return false;
    return rebuild([&](IRBuilderTy &builder) {
      return emitResign(builder, call->getArgOperand(0),
                        call->getArgOperand(1), call->getArgOperand(2),
                        call->getArgOperand(3), call->getArgOperand(4));
    });

  case Intrinsic::ptrauth_auth:
    if (!hasExpectedPrototype(call, VoidPtr, {VoidPtr, Key, Discriminator}))
      return false;
    return rebuild([&](IRBuilderTy &builder) {
      return emitAuth(builder, call->getArgOperand(0),
                      call->getArgOperand(1), call->getArgOperand(2));
    });

  case Intrinsic::ptrauth_strip:
    if (!hasExpectedPrototype(call, VoidPtr, {VoidPtr, Key}))
      return false;
    return rebuild([&](IRBuilderTy &builder) {
      return emitStrip(builder, call->getArgOperand(0),
                       call->getArgOperand(1));
    });

  case Intrinsic::ptrauth_blend:
    if (!hasExpectedPrototype(call, Discriminator,
                              {Discriminator, Discriminator}))
      return false;
    return rebuild([&](IRBuilderTy &builder) {
      return emitBlend(builder, call->getArgOperand(0),
                       call->getArgOperand(1));
    });

  case Intrinsic::ptrauth_sign_generic:
    if (!hasExpectedPrototype(call, IntPtr, {IntPtr, IntPtr}))
      return false;
    return rebuild([&](IRBuilderTy &builder) {
      return emitSignGeneric(builder, call->getArgOperand(0),
                             call->getArgOperand(1));
    });

  default:
    break;
  }

  return false;
}

bool SoftPointerAuth::transformInvoke(InvokeInst *call) {
  // Handle invokes with the llvm.ptrauth operand bundle attached.
  if (auto bundle = call->getOperandBundle(LLVMContext::OB_ptrauth)) {
    return transformPointerAuthCall(call, *bundle);
  }

  return false;
}

bool SoftPointerAuth::transformPointerAuthCall(CallBase *oldCall,
                                               const OperandBundleUse &bundle) {
  if (bundle.Inputs.size() != 2 ||
      !hasType(bundle.Inputs[0], Key) ||
      !hasType(bundle.Inputs[1], Discriminator))
    return false;

  IRBuilderTy builder(oldCall);

  // Authenticate the callee.
  Value *oldCallee = oldCall->getCalledOperand();
  Value *callee = builder.CreateBitCast(oldCallee, getType(VoidPtr));
  callee = emitAuth(builder, callee, bundle.Inputs[0], bundle.Inputs[1]);
  callee = builder.CreateBitCast(callee, oldCallee->getType());

  // Get the arguments.
  SmallVector<Value*, 8> args(oldCall->arg_begin(), oldCall->arg_end());

  // Get the operand bundles besides llvm.ptrauth (probably none).
  SmallVector<OperandBundleDef, 1> opBundles;
  for (unsigned i = 0, e = oldCall->getNumOperandBundles(); i != e; ++i) {
    auto bundle = oldCall->getOperandBundleAt(i);
    if (bundle.getTagID() != LLVMContext::OB_ptrauth) {
      opBundles.emplace_back(bundle);
    }
  }

  // Build the new instruction.
  CallBase *newCall;
  if (auto *oldInvoke = dyn_cast<InvokeInst>(oldCall)) {
    newCall = builder.CreateInvoke(oldInvoke->getFunctionType(), callee,
                                   oldInvoke->getNormalDest(),
                                   oldInvoke->getUnwindDest(),
                                   args, opBundles);
  } else {
    newCall =
        builder.CreateCall(oldCall->getFunctionType(), callee, args, opBundles);
  }

  // Copy mandatory attributes.
  newCall->setCallingConv(oldCall->getCallingConv());
  newCall->setAttributes(oldCall->getAttributes());

  // TODO: copy metadata?
  newCall->takeName(oldCall);

  // Destroy the old call.
  oldCall->replaceAllUsesWith(newCall);
  oldCall->eraseFromParent();

  return true;
}

/*****************************************************************************/
/**************************** Pass Manager Support ***************************/
/*****************************************************************************/

namespace {

class SoftPointerAuthLegacyPass : public ModulePass {
public:
  static char ID;
  SoftPointerAuthLegacyPass() : ModulePass(ID) {
    initializeSoftPointerAuthLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  StringRef getPassName() const override {
    return "Soft Pointer Auth Lowering";
  }
  bool runOnModule(Module &M) override { return Pass.runOnModule(M); }

private:
  SoftPointerAuth Pass;
};

} // end anonymous namespace

char SoftPointerAuthLegacyPass::ID = 0;
INITIALIZE_PASS(SoftPointerAuthLegacyPass, "soft-ptrauth",
                "Lower pointer authentication intrinsics for soft targets",
                false, false)

ModulePass *llvm::createSoftPointerAuthPass() {
  return new SoftPointerAuthLegacyPass();
}

PreservedAnalyses SoftPointerAuthPass::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  SoftPointerAuth Pass;
  if (!Pass.runOnModule(M))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}
