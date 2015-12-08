//===- unittests/IR/MetadataTest.cpp - Metadata unit tests ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

TEST(ContextAndReplaceableUsesTest, FromContext) {
  LLVMContext Context;
  ContextAndReplaceableUses CRU(Context);
  EXPECT_EQ(&Context, &CRU.getContext());
  EXPECT_FALSE(CRU.hasReplaceableUses());
  EXPECT_FALSE(CRU.getReplaceableUses());
}

TEST(ContextAndReplaceableUsesTest, FromReplaceableUses) {
  LLVMContext Context;
  ContextAndReplaceableUses CRU(make_unique<ReplaceableMetadataImpl>(Context));
  EXPECT_EQ(&Context, &CRU.getContext());
  EXPECT_TRUE(CRU.hasReplaceableUses());
  EXPECT_TRUE(CRU.getReplaceableUses());
}

TEST(ContextAndReplaceableUsesTest, makeReplaceable) {
  LLVMContext Context;
  ContextAndReplaceableUses CRU(Context);
  CRU.makeReplaceable(make_unique<ReplaceableMetadataImpl>(Context));
  EXPECT_EQ(&Context, &CRU.getContext());
  EXPECT_TRUE(CRU.hasReplaceableUses());
  EXPECT_TRUE(CRU.getReplaceableUses());
}

TEST(ContextAndReplaceableUsesTest, takeReplaceableUses) {
  LLVMContext Context;
  auto ReplaceableUses = make_unique<ReplaceableMetadataImpl>(Context);
  auto *Ptr = ReplaceableUses.get();
  ContextAndReplaceableUses CRU(std::move(ReplaceableUses));
  ReplaceableUses = CRU.takeReplaceableUses();
  EXPECT_EQ(&Context, &CRU.getContext());
  EXPECT_FALSE(CRU.hasReplaceableUses());
  EXPECT_FALSE(CRU.getReplaceableUses());
  EXPECT_EQ(Ptr, ReplaceableUses.get());
}

class MetadataTest : public testing::Test {
public:
  MetadataTest() : M("test", Context), Counter(0) {}

protected:
  LLVMContext Context;
  Module M;
  int Counter;

  MDNode *getNode() { return MDNode::get(Context, None); }
  MDNode *getNode(Metadata *MD) { return MDNode::get(Context, MD); }
  MDNode *getNode(Metadata *MD1, Metadata *MD2) {
    Metadata *MDs[] = {MD1, MD2};
    return MDNode::get(Context, MDs);
  }

  MDTuple *getTuple() { return MDTuple::getDistinct(Context, None); }
  DISubroutineType *getSubroutineType() {
    return DISubroutineType::getDistinct(Context, 0, getNode(nullptr));
  }
  DISubprogram *getSubprogram() {
    return DISubprogram::getDistinct(Context, nullptr, "", "", nullptr, 0,
                                     nullptr, false, false, 0, nullptr, 0, 0, 0,
                                     0);
  }
  DIScopeRef getSubprogramRef() { return getSubprogram()->getRef(); }
  DIFile *getFile() {
    return DIFile::getDistinct(Context, "file.c", "/path/to/dir");
  }
  DITypeRef getBasicType(StringRef Name) {
    return DIBasicType::get(Context, dwarf::DW_TAG_unspecified_type, Name)
        ->getRef();
  }
  DITypeRef getDerivedType() {
    return DIDerivedType::getDistinct(Context, dwarf::DW_TAG_pointer_type, "",
                                      nullptr, 0, nullptr,
                                      getBasicType("basictype"), 1, 2, 0, 0)
        ->getRef();
  }
  Constant *getConstant() {
    return ConstantInt::get(Type::getInt32Ty(Context), Counter++);
  }
  ConstantAsMetadata *getConstantAsMetadata() {
    return ConstantAsMetadata::get(getConstant());
  }
  DITypeRef getCompositeType() {
    return DICompositeType::getDistinct(
               Context, dwarf::DW_TAG_structure_type, "", nullptr, 0, nullptr,
               nullptr, 32, 32, 0, 0, nullptr, 0, nullptr, nullptr, "")
        ->getRef();
  }
  Function *getFunction(StringRef Name) {
    return cast<Function>(M.getOrInsertFunction(
        Name, FunctionType::get(Type::getVoidTy(Context), None, false)));
  }
};
typedef MetadataTest MDStringTest;

// Test that construction of MDString with different value produces different
// MDString objects, even with the same string pointer and nulls in the string.
TEST_F(MDStringTest, CreateDifferent) {
  char x[3] = { 'f', 0, 'A' };
  MDString *s1 = MDString::get(Context, StringRef(&x[0], 3));
  x[2] = 'B';
  MDString *s2 = MDString::get(Context, StringRef(&x[0], 3));
  EXPECT_NE(s1, s2);
}

// Test that creation of MDStrings with the same string contents produces the
// same MDString object, even with different pointers.
TEST_F(MDStringTest, CreateSame) {
  char x[4] = { 'a', 'b', 'c', 'X' };
  char y[4] = { 'a', 'b', 'c', 'Y' };

  MDString *s1 = MDString::get(Context, StringRef(&x[0], 3));
  MDString *s2 = MDString::get(Context, StringRef(&y[0], 3));
  EXPECT_EQ(s1, s2);
}

// Test that MDString prints out the string we fed it.
TEST_F(MDStringTest, PrintingSimple) {
  char *str = new char[13];
  strncpy(str, "testing 1 2 3", 13);
  MDString *s = MDString::get(Context, StringRef(str, 13));
  strncpy(str, "aaaaaaaaaaaaa", 13);
  delete[] str;

  std::string Str;
  raw_string_ostream oss(Str);
  s->print(oss);
  EXPECT_STREQ("!\"testing 1 2 3\"", oss.str().c_str());
}

// Test printing of MDString with non-printable characters.
TEST_F(MDStringTest, PrintingComplex) {
  char str[5] = {0, '\n', '"', '\\', (char)-1};
  MDString *s = MDString::get(Context, StringRef(str+0, 5));
  std::string Str;
  raw_string_ostream oss(Str);
  s->print(oss);
  EXPECT_STREQ("!\"\\00\\0A\\22\\5C\\FF\"", oss.str().c_str());
}

typedef MetadataTest MDNodeTest;

// Test the two constructors, and containing other Constants.
TEST_F(MDNodeTest, Simple) {
  char x[3] = { 'a', 'b', 'c' };
  char y[3] = { '1', '2', '3' };

  MDString *s1 = MDString::get(Context, StringRef(&x[0], 3));
  MDString *s2 = MDString::get(Context, StringRef(&y[0], 3));
  ConstantAsMetadata *CI = ConstantAsMetadata::get(
      ConstantInt::get(getGlobalContext(), APInt(8, 0)));

  std::vector<Metadata *> V;
  V.push_back(s1);
  V.push_back(CI);
  V.push_back(s2);

  MDNode *n1 = MDNode::get(Context, V);
  Metadata *const c1 = n1;
  MDNode *n2 = MDNode::get(Context, c1);
  Metadata *const c2 = n2;
  MDNode *n3 = MDNode::get(Context, V);
  MDNode *n4 = MDNode::getIfExists(Context, V);
  MDNode *n5 = MDNode::getIfExists(Context, c1);
  MDNode *n6 = MDNode::getIfExists(Context, c2);
  EXPECT_NE(n1, n2);
  EXPECT_EQ(n1, n3);
  EXPECT_EQ(n4, n1);
  EXPECT_EQ(n5, n2);
  EXPECT_EQ(n6, (Metadata *)nullptr);

  EXPECT_EQ(3u, n1->getNumOperands());
  EXPECT_EQ(s1, n1->getOperand(0));
  EXPECT_EQ(CI, n1->getOperand(1));
  EXPECT_EQ(s2, n1->getOperand(2));

  EXPECT_EQ(1u, n2->getNumOperands());
  EXPECT_EQ(n1, n2->getOperand(0));
}

TEST_F(MDNodeTest, Delete) {
  Constant *C = ConstantInt::get(Type::getInt32Ty(getGlobalContext()), 1);
  Instruction *I = new BitCastInst(C, Type::getInt32Ty(getGlobalContext()));

  Metadata *const V = LocalAsMetadata::get(I);
  MDNode *n = MDNode::get(Context, V);
  TrackingMDRef wvh(n);

  EXPECT_EQ(n, wvh);

  delete I;
}

TEST_F(MDNodeTest, SelfReference) {
  // !0 = !{!0}
  // !1 = !{!0}
  {
    auto Temp = MDNode::getTemporary(Context, None);
    Metadata *Args[] = {Temp.get()};
    MDNode *Self = MDNode::get(Context, Args);
    Self->replaceOperandWith(0, Self);
    ASSERT_EQ(Self, Self->getOperand(0));

    // Self-references should be distinct, so MDNode::get() should grab a
    // uniqued node that references Self, not Self.
    Args[0] = Self;
    MDNode *Ref1 = MDNode::get(Context, Args);
    MDNode *Ref2 = MDNode::get(Context, Args);
    EXPECT_NE(Self, Ref1);
    EXPECT_EQ(Ref1, Ref2);
  }

  // !0 = !{!0, !{}}
  // !1 = !{!0, !{}}
  {
    auto Temp = MDNode::getTemporary(Context, None);
    Metadata *Args[] = {Temp.get(), MDNode::get(Context, None)};
    MDNode *Self = MDNode::get(Context, Args);
    Self->replaceOperandWith(0, Self);
    ASSERT_EQ(Self, Self->getOperand(0));

    // Self-references should be distinct, so MDNode::get() should grab a
    // uniqued node that references Self, not Self itself.
    Args[0] = Self;
    MDNode *Ref1 = MDNode::get(Context, Args);
    MDNode *Ref2 = MDNode::get(Context, Args);
    EXPECT_NE(Self, Ref1);
    EXPECT_EQ(Ref1, Ref2);
  }
}

TEST_F(MDNodeTest, Print) {
  Constant *C = ConstantInt::get(Type::getInt32Ty(Context), 7);
  MDString *S = MDString::get(Context, "foo");
  MDNode *N0 = getNode();
  MDNode *N1 = getNode(N0);
  MDNode *N2 = getNode(N0, N1);

  Metadata *Args[] = {ConstantAsMetadata::get(C), S, nullptr, N0, N1, N2};
  MDNode *N = MDNode::get(Context, Args);

  std::string Expected;
  {
    raw_string_ostream OS(Expected);
    OS << "<" << (void *)N << "> = !{";
    C->printAsOperand(OS);
    OS << ", ";
    S->printAsOperand(OS);
    OS << ", null";
    MDNode *Nodes[] = {N0, N1, N2};
    for (auto *Node : Nodes)
      OS << ", <" << (void *)Node << ">";
    OS << "}";
  }

  std::string Actual;
  {
    raw_string_ostream OS(Actual);
    N->print(OS);
  }

  EXPECT_EQ(Expected, Actual);
}

#define EXPECT_PRINTER_EQ(EXPECTED, PRINT)                                     \
  do {                                                                         \
    std::string Actual_;                                                       \
    raw_string_ostream OS(Actual_);                                            \
    PRINT;                                                                     \
    OS.flush();                                                                \
    std::string Expected_(EXPECTED);                                           \
    EXPECT_EQ(Expected_, Actual_);                                             \
  } while (false)

TEST_F(MDNodeTest, PrintTemporary) {
  MDNode *Arg = getNode();
  TempMDNode Temp = MDNode::getTemporary(Context, Arg);
  MDNode *N = getNode(Temp.get());
  Module M("test", Context);
  NamedMDNode *NMD = M.getOrInsertNamedMetadata("named");
  NMD->addOperand(N);

  EXPECT_PRINTER_EQ("!0 = !{!1}", N->print(OS, &M));
  EXPECT_PRINTER_EQ("!1 = <temporary!> !{!2}", Temp->print(OS, &M));
  EXPECT_PRINTER_EQ("!2 = !{}", Arg->print(OS, &M));

  // Cleanup.
  Temp->replaceAllUsesWith(Arg);
}

TEST_F(MDNodeTest, PrintFromModule) {
  Constant *C = ConstantInt::get(Type::getInt32Ty(Context), 7);
  MDString *S = MDString::get(Context, "foo");
  MDNode *N0 = getNode();
  MDNode *N1 = getNode(N0);
  MDNode *N2 = getNode(N0, N1);

  Metadata *Args[] = {ConstantAsMetadata::get(C), S, nullptr, N0, N1, N2};
  MDNode *N = MDNode::get(Context, Args);
  Module M("test", Context);
  NamedMDNode *NMD = M.getOrInsertNamedMetadata("named");
  NMD->addOperand(N);

  std::string Expected;
  {
    raw_string_ostream OS(Expected);
    OS << "!0 = !{";
    C->printAsOperand(OS);
    OS << ", ";
    S->printAsOperand(OS);
    OS << ", null, !1, !2, !3}";
  }

  EXPECT_PRINTER_EQ(Expected, N->print(OS, &M));
}

TEST_F(MDNodeTest, PrintFromFunction) {
  Module M("test", Context);
  auto *FTy = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F0 = Function::Create(FTy, GlobalValue::ExternalLinkage, "F0", &M);
  auto *F1 = Function::Create(FTy, GlobalValue::ExternalLinkage, "F1", &M);
  auto *BB0 = BasicBlock::Create(Context, "entry", F0);
  auto *BB1 = BasicBlock::Create(Context, "entry", F1);
  auto *R0 = ReturnInst::Create(Context, BB0);
  auto *R1 = ReturnInst::Create(Context, BB1);
  auto *N0 = MDNode::getDistinct(Context, None);
  auto *N1 = MDNode::getDistinct(Context, None);
  R0->setMetadata("md", N0);
  R1->setMetadata("md", N1);

  EXPECT_PRINTER_EQ("!0 = distinct !{}", N0->print(OS, &M));
  EXPECT_PRINTER_EQ("!1 = distinct !{}", N1->print(OS, &M));

  ModuleSlotTracker MST(&M);
  EXPECT_PRINTER_EQ("!0 = distinct !{}", N0->print(OS, MST));
  EXPECT_PRINTER_EQ("!1 = distinct !{}", N1->print(OS, MST));
}

TEST_F(MDNodeTest, PrintFromMetadataAsValue) {
  Module M("test", Context);

  auto *Intrinsic =
      Function::Create(FunctionType::get(Type::getVoidTy(Context),
                                         Type::getMetadataTy(Context), false),
                       GlobalValue::ExternalLinkage, "llvm.intrinsic", &M);

  auto *FTy = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F0 = Function::Create(FTy, GlobalValue::ExternalLinkage, "F0", &M);
  auto *F1 = Function::Create(FTy, GlobalValue::ExternalLinkage, "F1", &M);
  auto *BB0 = BasicBlock::Create(Context, "entry", F0);
  auto *BB1 = BasicBlock::Create(Context, "entry", F1);
  auto *N0 = MDNode::getDistinct(Context, None);
  auto *N1 = MDNode::getDistinct(Context, None);
  auto *MAV0 = MetadataAsValue::get(Context, N0);
  auto *MAV1 = MetadataAsValue::get(Context, N1);
  CallInst::Create(Intrinsic, MAV0, "", BB0);
  CallInst::Create(Intrinsic, MAV1, "", BB1);

  EXPECT_PRINTER_EQ("!0 = distinct !{}", MAV0->print(OS));
  EXPECT_PRINTER_EQ("!1 = distinct !{}", MAV1->print(OS));
  EXPECT_PRINTER_EQ("!0", MAV0->printAsOperand(OS, false));
  EXPECT_PRINTER_EQ("!1", MAV1->printAsOperand(OS, false));
  EXPECT_PRINTER_EQ("metadata !0", MAV0->printAsOperand(OS, true));
  EXPECT_PRINTER_EQ("metadata !1", MAV1->printAsOperand(OS, true));

  ModuleSlotTracker MST(&M);
  EXPECT_PRINTER_EQ("!0 = distinct !{}", MAV0->print(OS, MST));
  EXPECT_PRINTER_EQ("!1 = distinct !{}", MAV1->print(OS, MST));
  EXPECT_PRINTER_EQ("!0", MAV0->printAsOperand(OS, false, MST));
  EXPECT_PRINTER_EQ("!1", MAV1->printAsOperand(OS, false, MST));
  EXPECT_PRINTER_EQ("metadata !0", MAV0->printAsOperand(OS, true, MST));
  EXPECT_PRINTER_EQ("metadata !1", MAV1->printAsOperand(OS, true, MST));
}
#undef EXPECT_PRINTER_EQ

TEST_F(MDNodeTest, NullOperand) {
  // metadata !{}
  MDNode *Empty = MDNode::get(Context, None);

  // metadata !{metadata !{}}
  Metadata *Ops[] = {Empty};
  MDNode *N = MDNode::get(Context, Ops);
  ASSERT_EQ(Empty, N->getOperand(0));

  // metadata !{metadata !{}} => metadata !{null}
  N->replaceOperandWith(0, nullptr);
  ASSERT_EQ(nullptr, N->getOperand(0));

  // metadata !{null}
  Ops[0] = nullptr;
  MDNode *NullOp = MDNode::get(Context, Ops);
  ASSERT_EQ(nullptr, NullOp->getOperand(0));
  EXPECT_EQ(N, NullOp);
}

TEST_F(MDNodeTest, DistinctOnUniquingCollision) {
  // !{}
  MDNode *Empty = MDNode::get(Context, None);
  ASSERT_TRUE(Empty->isResolved());
  EXPECT_FALSE(Empty->isDistinct());

  // !{!{}}
  Metadata *Wrapped1Ops[] = {Empty};
  MDNode *Wrapped1 = MDNode::get(Context, Wrapped1Ops);
  ASSERT_EQ(Empty, Wrapped1->getOperand(0));
  ASSERT_TRUE(Wrapped1->isResolved());
  EXPECT_FALSE(Wrapped1->isDistinct());

  // !{!{!{}}}
  Metadata *Wrapped2Ops[] = {Wrapped1};
  MDNode *Wrapped2 = MDNode::get(Context, Wrapped2Ops);
  ASSERT_EQ(Wrapped1, Wrapped2->getOperand(0));
  ASSERT_TRUE(Wrapped2->isResolved());
  EXPECT_FALSE(Wrapped2->isDistinct());

  // !{!{!{}}} => !{!{}}
  Wrapped2->replaceOperandWith(0, Empty);
  ASSERT_EQ(Empty, Wrapped2->getOperand(0));
  EXPECT_TRUE(Wrapped2->isDistinct());
  EXPECT_FALSE(Wrapped1->isDistinct());
}

TEST_F(MDNodeTest, getDistinct) {
  // !{}
  MDNode *Empty = MDNode::get(Context, None);
  ASSERT_TRUE(Empty->isResolved());
  ASSERT_FALSE(Empty->isDistinct());
  ASSERT_EQ(Empty, MDNode::get(Context, None));

  // distinct !{}
  MDNode *Distinct1 = MDNode::getDistinct(Context, None);
  MDNode *Distinct2 = MDNode::getDistinct(Context, None);
  EXPECT_TRUE(Distinct1->isResolved());
  EXPECT_TRUE(Distinct2->isDistinct());
  EXPECT_NE(Empty, Distinct1);
  EXPECT_NE(Empty, Distinct2);
  EXPECT_NE(Distinct1, Distinct2);

  // !{}
  ASSERT_EQ(Empty, MDNode::get(Context, None));
}

TEST_F(MDNodeTest, isUniqued) {
  MDNode *U = MDTuple::get(Context, None);
  MDNode *D = MDTuple::getDistinct(Context, None);
  auto T = MDTuple::getTemporary(Context, None);
  EXPECT_TRUE(U->isUniqued());
  EXPECT_FALSE(D->isUniqued());
  EXPECT_FALSE(T->isUniqued());
}

TEST_F(MDNodeTest, isDistinct) {
  MDNode *U = MDTuple::get(Context, None);
  MDNode *D = MDTuple::getDistinct(Context, None);
  auto T = MDTuple::getTemporary(Context, None);
  EXPECT_FALSE(U->isDistinct());
  EXPECT_TRUE(D->isDistinct());
  EXPECT_FALSE(T->isDistinct());
}

TEST_F(MDNodeTest, isTemporary) {
  MDNode *U = MDTuple::get(Context, None);
  MDNode *D = MDTuple::getDistinct(Context, None);
  auto T = MDTuple::getTemporary(Context, None);
  EXPECT_FALSE(U->isTemporary());
  EXPECT_FALSE(D->isTemporary());
  EXPECT_TRUE(T->isTemporary());
}

TEST_F(MDNodeTest, getDistinctWithUnresolvedOperands) {
  // temporary !{}
  auto Temp = MDTuple::getTemporary(Context, None);
  ASSERT_FALSE(Temp->isResolved());

  // distinct !{temporary !{}}
  Metadata *Ops[] = {Temp.get()};
  MDNode *Distinct = MDNode::getDistinct(Context, Ops);
  EXPECT_TRUE(Distinct->isResolved());
  EXPECT_EQ(Temp.get(), Distinct->getOperand(0));

  // temporary !{} => !{}
  MDNode *Empty = MDNode::get(Context, None);
  Temp->replaceAllUsesWith(Empty);
  EXPECT_EQ(Empty, Distinct->getOperand(0));
}

TEST_F(MDNodeTest, handleChangedOperandRecursion) {
  // !0 = !{}
  MDNode *N0 = MDNode::get(Context, None);

  // !1 = !{!3, null}
  auto Temp3 = MDTuple::getTemporary(Context, None);
  Metadata *Ops1[] = {Temp3.get(), nullptr};
  MDNode *N1 = MDNode::get(Context, Ops1);

  // !2 = !{!3, !0}
  Metadata *Ops2[] = {Temp3.get(), N0};
  MDNode *N2 = MDNode::get(Context, Ops2);

  // !3 = !{!2}
  Metadata *Ops3[] = {N2};
  MDNode *N3 = MDNode::get(Context, Ops3);
  Temp3->replaceAllUsesWith(N3);

  // !4 = !{!1}
  Metadata *Ops4[] = {N1};
  MDNode *N4 = MDNode::get(Context, Ops4);

  // Confirm that the cycle prevented RAUW from getting dropped.
  EXPECT_TRUE(N0->isResolved());
  EXPECT_FALSE(N1->isResolved());
  EXPECT_FALSE(N2->isResolved());
  EXPECT_FALSE(N3->isResolved());
  EXPECT_FALSE(N4->isResolved());

  // Create a couple of distinct nodes to observe what's going on.
  //
  // !5 = distinct !{!2}
  // !6 = distinct !{!3}
  Metadata *Ops5[] = {N2};
  MDNode *N5 = MDNode::getDistinct(Context, Ops5);
  Metadata *Ops6[] = {N3};
  MDNode *N6 = MDNode::getDistinct(Context, Ops6);

  // Mutate !2 to look like !1, causing a uniquing collision (and an RAUW).
  // This will ripple up, with !3 colliding with !4, and RAUWing.  Since !2
  // references !3, this can cause a re-entry of handleChangedOperand() when !3
  // is not ready for it.
  //
  // !2->replaceOperandWith(1, nullptr)
  // !2: !{!3, !0} => !{!3, null}
  // !2->replaceAllUsesWith(!1)
  // !3: !{!2] => !{!1}
  // !3->replaceAllUsesWith(!4)
  N2->replaceOperandWith(1, nullptr);

  // If all has gone well, N2 and N3 will have been RAUW'ed and deleted from
  // under us.  Just check that the other nodes are sane.
  //
  // !1 = !{!4, null}
  // !4 = !{!1}
  // !5 = distinct !{!1}
  // !6 = distinct !{!4}
  EXPECT_EQ(N4, N1->getOperand(0));
  EXPECT_EQ(N1, N4->getOperand(0));
  EXPECT_EQ(N1, N5->getOperand(0));
  EXPECT_EQ(N4, N6->getOperand(0));
}

TEST_F(MDNodeTest, replaceResolvedOperand) {
  // Check code for replacing one resolved operand with another.  If doing this
  // directly (via replaceOperandWith()) becomes illegal, change the operand to
  // a global value that gets RAUW'ed.
  //
  // Use a temporary node to keep N from being resolved.
  auto Temp = MDTuple::getTemporary(Context, None);
  Metadata *Ops[] = {nullptr, Temp.get()};

  MDNode *Empty = MDTuple::get(Context, ArrayRef<Metadata *>());
  MDNode *N = MDTuple::get(Context, Ops);
  EXPECT_EQ(nullptr, N->getOperand(0));
  ASSERT_FALSE(N->isResolved());

  // Check code for replacing resolved nodes.
  N->replaceOperandWith(0, Empty);
  EXPECT_EQ(Empty, N->getOperand(0));

  // Check code for adding another unresolved operand.
  N->replaceOperandWith(0, Temp.get());
  EXPECT_EQ(Temp.get(), N->getOperand(0));

  // Remove the references to Temp; required for teardown.
  Temp->replaceAllUsesWith(nullptr);
}

TEST_F(MDNodeTest, replaceWithUniqued) {
  auto *Empty = MDTuple::get(Context, None);
  MDTuple *FirstUniqued;
  {
    Metadata *Ops[] = {Empty};
    auto Temp = MDTuple::getTemporary(Context, Ops);
    EXPECT_TRUE(Temp->isTemporary());

    // Don't expect a collision.
    auto *Current = Temp.get();
    FirstUniqued = MDNode::replaceWithUniqued(std::move(Temp));
    EXPECT_TRUE(FirstUniqued->isUniqued());
    EXPECT_TRUE(FirstUniqued->isResolved());
    EXPECT_EQ(Current, FirstUniqued);
  }
  {
    Metadata *Ops[] = {Empty};
    auto Temp = MDTuple::getTemporary(Context, Ops);
    EXPECT_TRUE(Temp->isTemporary());

    // Should collide with Uniqued above this time.
    auto *Uniqued = MDNode::replaceWithUniqued(std::move(Temp));
    EXPECT_TRUE(Uniqued->isUniqued());
    EXPECT_TRUE(Uniqued->isResolved());
    EXPECT_EQ(FirstUniqued, Uniqued);
  }
  {
    auto Unresolved = MDTuple::getTemporary(Context, None);
    Metadata *Ops[] = {Unresolved.get()};
    auto Temp = MDTuple::getTemporary(Context, Ops);
    EXPECT_TRUE(Temp->isTemporary());

    // Shouldn't be resolved.
    auto *Uniqued = MDNode::replaceWithUniqued(std::move(Temp));
    EXPECT_TRUE(Uniqued->isUniqued());
    EXPECT_FALSE(Uniqued->isResolved());

    // Should be a different node.
    EXPECT_NE(FirstUniqued, Uniqued);

    // Should resolve when we update its node (note: be careful to avoid a
    // collision with any other nodes above).
    Uniqued->replaceOperandWith(0, nullptr);
    EXPECT_TRUE(Uniqued->isResolved());
  }
}

TEST_F(MDNodeTest, replaceWithUniquedResolvingOperand) {
  // temp !{}
  MDTuple *Op = MDTuple::getTemporary(Context, None).release();
  EXPECT_FALSE(Op->isResolved());

  // temp !{temp !{}}
  Metadata *Ops[] = {Op};
  MDTuple *N = MDTuple::getTemporary(Context, Ops).release();
  EXPECT_FALSE(N->isResolved());

  // temp !{temp !{}} => !{temp !{}}
  ASSERT_EQ(N, MDNode::replaceWithUniqued(TempMDTuple(N)));
  EXPECT_FALSE(N->isResolved());

  // !{temp !{}} => !{!{}}
  ASSERT_EQ(Op, MDNode::replaceWithUniqued(TempMDTuple(Op)));
  EXPECT_TRUE(Op->isResolved());
  EXPECT_TRUE(N->isResolved());
}

TEST_F(MDNodeTest, replaceWithUniquedChangingOperand) {
  // i1* @GV
  Type *Ty = Type::getInt1PtrTy(Context);
  std::unique_ptr<GlobalVariable> GV(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  ConstantAsMetadata *Op = ConstantAsMetadata::get(GV.get());

  // temp !{i1* @GV}
  Metadata *Ops[] = {Op};
  MDTuple *N = MDTuple::getTemporary(Context, Ops).release();

  // temp !{i1* @GV} => !{i1* @GV}
  ASSERT_EQ(N, MDNode::replaceWithUniqued(TempMDTuple(N)));
  ASSERT_TRUE(N->isUniqued());

  // !{i1* @GV} => !{null}
  GV.reset();
  ASSERT_TRUE(N->isUniqued());
  Metadata *NullOps[] = {nullptr};
  ASSERT_EQ(N, MDTuple::get(Context, NullOps));
}

TEST_F(MDNodeTest, replaceWithDistinct) {
  {
    auto *Empty = MDTuple::get(Context, None);
    Metadata *Ops[] = {Empty};
    auto Temp = MDTuple::getTemporary(Context, Ops);
    EXPECT_TRUE(Temp->isTemporary());

    // Don't expect a collision.
    auto *Current = Temp.get();
    auto *Distinct = MDNode::replaceWithDistinct(std::move(Temp));
    EXPECT_TRUE(Distinct->isDistinct());
    EXPECT_TRUE(Distinct->isResolved());
    EXPECT_EQ(Current, Distinct);
  }
  {
    auto Unresolved = MDTuple::getTemporary(Context, None);
    Metadata *Ops[] = {Unresolved.get()};
    auto Temp = MDTuple::getTemporary(Context, Ops);
    EXPECT_TRUE(Temp->isTemporary());

    // Don't expect a collision.
    auto *Current = Temp.get();
    auto *Distinct = MDNode::replaceWithDistinct(std::move(Temp));
    EXPECT_TRUE(Distinct->isDistinct());
    EXPECT_TRUE(Distinct->isResolved());
    EXPECT_EQ(Current, Distinct);

    // Cleanup; required for teardown.
    Unresolved->replaceAllUsesWith(nullptr);
  }
}

TEST_F(MDNodeTest, replaceWithPermanent) {
  Metadata *Ops[] = {nullptr};
  auto Temp = MDTuple::getTemporary(Context, Ops);
  auto *T = Temp.get();

  // U is a normal, uniqued node that references T.
  auto *U = MDTuple::get(Context, T);
  EXPECT_TRUE(U->isUniqued());

  // Make Temp self-referencing.
  Temp->replaceOperandWith(0, T);

  // Try to uniquify Temp.  This should, despite the name in the API, give a
  // 'distinct' node, since self-references aren't allowed to be uniqued.
  //
  // Since it's distinct, N should have the same address as when it was a
  // temporary (i.e., be equal to T not U).
  auto *N = MDNode::replaceWithPermanent(std::move(Temp));
  EXPECT_EQ(N, T);
  EXPECT_TRUE(N->isDistinct());

  // U should be the canonical unique node with N as the argument.
  EXPECT_EQ(U, MDTuple::get(Context, N));
  EXPECT_TRUE(U->isUniqued());

  // This temporary should collide with U when replaced, but it should still be
  // uniqued.
  EXPECT_EQ(U, MDNode::replaceWithPermanent(MDTuple::getTemporary(Context, N)));
  EXPECT_TRUE(U->isUniqued());

  // This temporary should become a new uniqued node.
  auto Temp2 = MDTuple::getTemporary(Context, U);
  auto *V = Temp2.get();
  EXPECT_EQ(V, MDNode::replaceWithPermanent(std::move(Temp2)));
  EXPECT_TRUE(V->isUniqued());
  EXPECT_EQ(U, V->getOperand(0));
}

TEST_F(MDNodeTest, deleteTemporaryWithTrackingRef) {
  TrackingMDRef Ref;
  EXPECT_EQ(nullptr, Ref.get());
  {
    auto Temp = MDTuple::getTemporary(Context, None);
    Ref.reset(Temp.get());
    EXPECT_EQ(Temp.get(), Ref.get());
  }
  EXPECT_EQ(nullptr, Ref.get());
}

typedef MetadataTest DILocationTest;

TEST_F(DILocationTest, Overflow) {
  DISubprogram *N = getSubprogram();
  {
    DILocation *L = DILocation::get(Context, 2, 7, N);
    EXPECT_EQ(2u, L->getLine());
    EXPECT_EQ(7u, L->getColumn());
  }
  unsigned U16 = 1u << 16;
  {
    DILocation *L = DILocation::get(Context, UINT32_MAX, U16 - 1, N);
    EXPECT_EQ(UINT32_MAX, L->getLine());
    EXPECT_EQ(U16 - 1, L->getColumn());
  }
  {
    DILocation *L = DILocation::get(Context, UINT32_MAX, U16, N);
    EXPECT_EQ(UINT32_MAX, L->getLine());
    EXPECT_EQ(0u, L->getColumn());
  }
  {
    DILocation *L = DILocation::get(Context, UINT32_MAX, U16 + 1, N);
    EXPECT_EQ(UINT32_MAX, L->getLine());
    EXPECT_EQ(0u, L->getColumn());
  }
}

TEST_F(DILocationTest, getDistinct) {
  MDNode *N = getSubprogram();
  DILocation *L0 = DILocation::getDistinct(Context, 2, 7, N);
  EXPECT_TRUE(L0->isDistinct());
  DILocation *L1 = DILocation::get(Context, 2, 7, N);
  EXPECT_FALSE(L1->isDistinct());
  EXPECT_EQ(L1, DILocation::get(Context, 2, 7, N));
}

TEST_F(DILocationTest, getTemporary) {
  MDNode *N = MDNode::get(Context, None);
  auto L = DILocation::getTemporary(Context, 2, 7, N);
  EXPECT_TRUE(L->isTemporary());
  EXPECT_FALSE(L->isResolved());
}

TEST_F(DILocationTest, cloneTemporary) {
  MDNode *N = MDNode::get(Context, None);
  auto L = DILocation::getTemporary(Context, 2, 7, N);
  EXPECT_TRUE(L->isTemporary());
  auto L2 = L->clone();
  EXPECT_TRUE(L2->isTemporary());
}

typedef MetadataTest GenericDINodeTest;

TEST_F(GenericDINodeTest, get) {
  StringRef Header = "header";
  auto *Empty = MDNode::get(Context, None);
  Metadata *Ops1[] = {Empty};
  auto *N = GenericDINode::get(Context, 15, Header, Ops1);
  EXPECT_EQ(15u, N->getTag());
  EXPECT_EQ(2u, N->getNumOperands());
  EXPECT_EQ(Header, N->getHeader());
  EXPECT_EQ(MDString::get(Context, Header), N->getOperand(0));
  EXPECT_EQ(1u, N->getNumDwarfOperands());
  EXPECT_EQ(Empty, N->getDwarfOperand(0));
  EXPECT_EQ(Empty, N->getOperand(1));
  ASSERT_TRUE(N->isUniqued());

  EXPECT_EQ(N, GenericDINode::get(Context, 15, Header, Ops1));

  N->replaceOperandWith(1, nullptr);
  EXPECT_EQ(15u, N->getTag());
  EXPECT_EQ(Header, N->getHeader());
  EXPECT_EQ(nullptr, N->getDwarfOperand(0));
  ASSERT_TRUE(N->isUniqued());

  Metadata *Ops2[] = {nullptr};
  EXPECT_EQ(N, GenericDINode::get(Context, 15, Header, Ops2));

  N->replaceDwarfOperandWith(0, Empty);
  EXPECT_EQ(15u, N->getTag());
  EXPECT_EQ(Header, N->getHeader());
  EXPECT_EQ(Empty, N->getDwarfOperand(0));
  ASSERT_TRUE(N->isUniqued());
  EXPECT_EQ(N, GenericDINode::get(Context, 15, Header, Ops1));

  TempGenericDINode Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(GenericDINodeTest, getEmptyHeader) {
  // Canonicalize !"" to null.
  auto *N = GenericDINode::get(Context, 15, StringRef(), None);
  EXPECT_EQ(StringRef(), N->getHeader());
  EXPECT_EQ(nullptr, N->getOperand(0));
}

typedef MetadataTest DISubrangeTest;

TEST_F(DISubrangeTest, get) {
  auto *N = DISubrange::get(Context, 5, 7);
  EXPECT_EQ(dwarf::DW_TAG_subrange_type, N->getTag());
  EXPECT_EQ(5, N->getCount());
  EXPECT_EQ(7, N->getLowerBound());
  EXPECT_EQ(N, DISubrange::get(Context, 5, 7));
  EXPECT_EQ(DISubrange::get(Context, 5, 0), DISubrange::get(Context, 5));

  TempDISubrange Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DISubrangeTest, getEmptyArray) {
  auto *N = DISubrange::get(Context, -1, 0);
  EXPECT_EQ(dwarf::DW_TAG_subrange_type, N->getTag());
  EXPECT_EQ(-1, N->getCount());
  EXPECT_EQ(0, N->getLowerBound());
  EXPECT_EQ(N, DISubrange::get(Context, -1, 0));
}

typedef MetadataTest DIEnumeratorTest;

TEST_F(DIEnumeratorTest, get) {
  auto *N = DIEnumerator::get(Context, 7, "name");
  EXPECT_EQ(dwarf::DW_TAG_enumerator, N->getTag());
  EXPECT_EQ(7, N->getValue());
  EXPECT_EQ("name", N->getName());
  EXPECT_EQ(N, DIEnumerator::get(Context, 7, "name"));

  EXPECT_NE(N, DIEnumerator::get(Context, 8, "name"));
  EXPECT_NE(N, DIEnumerator::get(Context, 7, "nam"));

  TempDIEnumerator Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DIBasicTypeTest;

TEST_F(DIBasicTypeTest, get) {
  auto *N =
      DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33, 26, 7);
  EXPECT_EQ(dwarf::DW_TAG_base_type, N->getTag());
  EXPECT_EQ("special", N->getName());
  EXPECT_EQ(33u, N->getSizeInBits());
  EXPECT_EQ(26u, N->getAlignInBits());
  EXPECT_EQ(7u, N->getEncoding());
  EXPECT_EQ(0u, N->getLine());
  EXPECT_EQ(N, DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33,
                                26, 7));

  EXPECT_NE(N, DIBasicType::get(Context, dwarf::DW_TAG_unspecified_type,
                                "special", 33, 26, 7));
  EXPECT_NE(N,
            DIBasicType::get(Context, dwarf::DW_TAG_base_type, "s", 33, 26, 7));
  EXPECT_NE(N, DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 32,
                                26, 7));
  EXPECT_NE(N, DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33,
                                25, 7));
  EXPECT_NE(N, DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33,
                                26, 6));

  TempDIBasicType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DIBasicTypeTest, getWithLargeValues) {
  auto *N = DIBasicType::get(Context, dwarf::DW_TAG_base_type, "special",
                             UINT64_MAX, UINT64_MAX - 1, 7);
  EXPECT_EQ(UINT64_MAX, N->getSizeInBits());
  EXPECT_EQ(UINT64_MAX - 1, N->getAlignInBits());
}

TEST_F(DIBasicTypeTest, getUnspecified) {
  auto *N =
      DIBasicType::get(Context, dwarf::DW_TAG_unspecified_type, "unspecified");
  EXPECT_EQ(dwarf::DW_TAG_unspecified_type, N->getTag());
  EXPECT_EQ("unspecified", N->getName());
  EXPECT_EQ(0u, N->getSizeInBits());
  EXPECT_EQ(0u, N->getAlignInBits());
  EXPECT_EQ(0u, N->getEncoding());
  EXPECT_EQ(0u, N->getLine());
}

typedef MetadataTest DITypeTest;

TEST_F(DITypeTest, clone) {
  // Check that DIType has a specialized clone that returns TempDIType.
  DIType *N = DIBasicType::get(Context, dwarf::DW_TAG_base_type, "int", 32, 32,
                               dwarf::DW_ATE_signed);

  TempDIType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DITypeTest, setFlags) {
  // void (void)
  Metadata *TypesOps[] = {nullptr};
  Metadata *Types = MDTuple::get(Context, TypesOps);

  DIType *D = DISubroutineType::getDistinct(Context, 0u, Types);
  EXPECT_EQ(0u, D->getFlags());
  D->setFlags(DINode::FlagRValueReference);
  EXPECT_EQ(DINode::FlagRValueReference, D->getFlags());
  D->setFlags(0u);
  EXPECT_EQ(0u, D->getFlags());

  TempDIType T = DISubroutineType::getTemporary(Context, 0u, Types);
  EXPECT_EQ(0u, T->getFlags());
  T->setFlags(DINode::FlagRValueReference);
  EXPECT_EQ(DINode::FlagRValueReference, T->getFlags());
  T->setFlags(0u);
  EXPECT_EQ(0u, T->getFlags());
}

typedef MetadataTest DIDerivedTypeTest;

TEST_F(DIDerivedTypeTest, get) {
  DIFile *File = getFile();
  DIScopeRef Scope = getSubprogramRef();
  DITypeRef BaseType = getBasicType("basic");
  MDTuple *ExtraData = getTuple();

  auto *N = DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type, "something",
                               File, 1, Scope, BaseType, 2, 3, 4, 5, ExtraData);
  EXPECT_EQ(dwarf::DW_TAG_pointer_type, N->getTag());
  EXPECT_EQ("something", N->getName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(1u, N->getLine());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(BaseType, N->getBaseType());
  EXPECT_EQ(2u, N->getSizeInBits());
  EXPECT_EQ(3u, N->getAlignInBits());
  EXPECT_EQ(4u, N->getOffsetInBits());
  EXPECT_EQ(5u, N->getFlags());
  EXPECT_EQ(ExtraData, N->getExtraData());
  EXPECT_EQ(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  4, 5, ExtraData));

  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_reference_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  4, 5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type, "else",
                                  File, 1, Scope, BaseType, 2, 3, 4, 5,
                                  ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", getFile(), 1, Scope, BaseType, 2,
                                  3, 4, 5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 2, Scope, BaseType, 2, 3,
                                  4, 5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, getSubprogramRef(),
                                  BaseType, 2, 3, 4, 5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(
                   Context, dwarf::DW_TAG_pointer_type, "something", File, 1,
                   Scope, getBasicType("basic2"), 2, 3, 4, 5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 3, 3,
                                  4, 5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 2,
                                  4, 5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  5, 5, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  4, 4, ExtraData));
  EXPECT_NE(N, DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  4, 5, getTuple()));

  TempDIDerivedType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DIDerivedTypeTest, getWithLargeValues) {
  DIFile *File = getFile();
  DIScopeRef Scope = getSubprogramRef();
  DITypeRef BaseType = getBasicType("basic");
  MDTuple *ExtraData = getTuple();

  auto *N = DIDerivedType::get(Context, dwarf::DW_TAG_pointer_type, "something",
                               File, 1, Scope, BaseType, UINT64_MAX,
                               UINT64_MAX - 1, UINT64_MAX - 2, 5, ExtraData);
  EXPECT_EQ(UINT64_MAX, N->getSizeInBits());
  EXPECT_EQ(UINT64_MAX - 1, N->getAlignInBits());
  EXPECT_EQ(UINT64_MAX - 2, N->getOffsetInBits());
}

typedef MetadataTest DICompositeTypeTest;

TEST_F(DICompositeTypeTest, get) {
  unsigned Tag = dwarf::DW_TAG_structure_type;
  StringRef Name = "some name";
  DIFile *File = getFile();
  unsigned Line = 1;
  DIScopeRef Scope = getSubprogramRef();
  DITypeRef BaseType = getCompositeType();
  uint64_t SizeInBits = 2;
  uint64_t AlignInBits = 3;
  uint64_t OffsetInBits = 4;
  unsigned Flags = 5;
  MDTuple *Elements = getTuple();
  unsigned RuntimeLang = 6;
  DITypeRef VTableHolder = getCompositeType();
  MDTuple *TemplateParams = getTuple();
  StringRef Identifier = "some id";

  auto *N = DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                 BaseType, SizeInBits, AlignInBits,
                                 OffsetInBits, Flags, Elements, RuntimeLang,
                                 VTableHolder, TemplateParams, Identifier);
  EXPECT_EQ(Tag, N->getTag());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(BaseType, N->getBaseType());
  EXPECT_EQ(SizeInBits, N->getSizeInBits());
  EXPECT_EQ(AlignInBits, N->getAlignInBits());
  EXPECT_EQ(OffsetInBits, N->getOffsetInBits());
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(Elements, N->getElements().get());
  EXPECT_EQ(RuntimeLang, N->getRuntimeLang());
  EXPECT_EQ(VTableHolder, N->getVTableHolder());
  EXPECT_EQ(TemplateParams, N->getTemplateParams().get());
  EXPECT_EQ(Identifier, N->getIdentifier());

  EXPECT_EQ(N, DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));

  EXPECT_NE(N, DICompositeType::get(Context, Tag + 1, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, "abc", File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, Name, getFile(), Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, Name, File, Line + 1, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, getSubprogramRef(), BaseType,
                   SizeInBits, AlignInBits, OffsetInBits, Flags, Elements,
                   RuntimeLang, VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, Scope, getBasicType("other"),
                   SizeInBits, AlignInBits, OffsetInBits, Flags, Elements,
                   RuntimeLang, VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits + 1, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits + 1,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits + 1, Flags, Elements, RuntimeLang,
                   VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits, Flags + 1, Elements, RuntimeLang,
                   VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits, Flags, getTuple(), RuntimeLang,
                   VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang + 1,
                   VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang,
                   getCompositeType(), TemplateParams, Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, getTuple(), Identifier));
  EXPECT_NE(N, DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, "other"));

  // Be sure that missing identifiers get null pointers.
  EXPECT_FALSE(DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, "")
                   ->getRawIdentifier());
  EXPECT_FALSE(DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams)
                   ->getRawIdentifier());

  TempDICompositeType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DICompositeTypeTest, getWithLargeValues) {
  unsigned Tag = dwarf::DW_TAG_structure_type;
  StringRef Name = "some name";
  DIFile *File = getFile();
  unsigned Line = 1;
  DIScopeRef Scope = getSubprogramRef();
  DITypeRef BaseType = getCompositeType();
  uint64_t SizeInBits = UINT64_MAX;
  uint64_t AlignInBits = UINT64_MAX - 1;
  uint64_t OffsetInBits = UINT64_MAX - 2;
  unsigned Flags = 5;
  MDTuple *Elements = getTuple();
  unsigned RuntimeLang = 6;
  DITypeRef VTableHolder = getCompositeType();
  MDTuple *TemplateParams = getTuple();
  StringRef Identifier = "some id";

  auto *N = DICompositeType::get(Context, Tag, Name, File, Line, Scope,
                                 BaseType, SizeInBits, AlignInBits,
                                 OffsetInBits, Flags, Elements, RuntimeLang,
                                 VTableHolder, TemplateParams, Identifier);
  EXPECT_EQ(SizeInBits, N->getSizeInBits());
  EXPECT_EQ(AlignInBits, N->getAlignInBits());
  EXPECT_EQ(OffsetInBits, N->getOffsetInBits());
}

TEST_F(DICompositeTypeTest, replaceOperands) {
  unsigned Tag = dwarf::DW_TAG_structure_type;
  StringRef Name = "some name";
  DIFile *File = getFile();
  unsigned Line = 1;
  DIScopeRef Scope = getSubprogramRef();
  DITypeRef BaseType = getCompositeType();
  uint64_t SizeInBits = 2;
  uint64_t AlignInBits = 3;
  uint64_t OffsetInBits = 4;
  unsigned Flags = 5;
  unsigned RuntimeLang = 6;
  StringRef Identifier = "some id";

  auto *N = DICompositeType::get(
      Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits, AlignInBits,
      OffsetInBits, Flags, nullptr, RuntimeLang, nullptr, nullptr, Identifier);

  auto *Elements = MDTuple::getDistinct(Context, None);
  EXPECT_EQ(nullptr, N->getElements().get());
  N->replaceElements(Elements);
  EXPECT_EQ(Elements, N->getElements().get());
  N->replaceElements(nullptr);
  EXPECT_EQ(nullptr, N->getElements().get());

  DITypeRef VTableHolder = getCompositeType();
  EXPECT_EQ(nullptr, N->getVTableHolder());
  N->replaceVTableHolder(VTableHolder);
  EXPECT_EQ(VTableHolder, N->getVTableHolder());
  N->replaceVTableHolder(nullptr);
  EXPECT_EQ(nullptr, N->getVTableHolder());

  auto *TemplateParams = MDTuple::getDistinct(Context, None);
  EXPECT_EQ(nullptr, N->getTemplateParams().get());
  N->replaceTemplateParams(TemplateParams);
  EXPECT_EQ(TemplateParams, N->getTemplateParams().get());
  N->replaceTemplateParams(nullptr);
  EXPECT_EQ(nullptr, N->getTemplateParams().get());
}

typedef MetadataTest DISubroutineTypeTest;

TEST_F(DISubroutineTypeTest, get) {
  unsigned Flags = 1;
  MDTuple *TypeArray = getTuple();

  auto *N = DISubroutineType::get(Context, Flags, TypeArray);
  EXPECT_EQ(dwarf::DW_TAG_subroutine_type, N->getTag());
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(TypeArray, N->getTypeArray().get());
  EXPECT_EQ(N, DISubroutineType::get(Context, Flags, TypeArray));

  EXPECT_NE(N, DISubroutineType::get(Context, Flags + 1, TypeArray));
  EXPECT_NE(N, DISubroutineType::get(Context, Flags, getTuple()));

  TempDISubroutineType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));

  // Test always-empty operands.
  EXPECT_EQ(nullptr, N->getScope());
  EXPECT_EQ(nullptr, N->getFile());
  EXPECT_EQ("", N->getName());
}

typedef MetadataTest DIFileTest;

TEST_F(DIFileTest, get) {
  StringRef Filename = "file";
  StringRef Directory = "dir";
  auto *N = DIFile::get(Context, Filename, Directory);

  EXPECT_EQ(dwarf::DW_TAG_file_type, N->getTag());
  EXPECT_EQ(Filename, N->getFilename());
  EXPECT_EQ(Directory, N->getDirectory());
  EXPECT_EQ(N, DIFile::get(Context, Filename, Directory));

  EXPECT_NE(N, DIFile::get(Context, "other", Directory));
  EXPECT_NE(N, DIFile::get(Context, Filename, "other"));

  TempDIFile Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DIFileTest, ScopeGetFile) {
  // Ensure that DIScope::getFile() returns itself.
  DIScope *N = DIFile::get(Context, "file", "dir");
  EXPECT_EQ(N, N->getFile());
}

typedef MetadataTest DICompileUnitTest;

TEST_F(DICompileUnitTest, get) {
  unsigned SourceLanguage = 1;
  DIFile *File = getFile();
  StringRef Producer = "some producer";
  bool IsOptimized = false;
  StringRef Flags = "flag after flag";
  unsigned RuntimeVersion = 2;
  StringRef SplitDebugFilename = "another/file";
  unsigned EmissionKind = 3;
  MDTuple *EnumTypes = getTuple();
  MDTuple *RetainedTypes = getTuple();
  MDTuple *Subprograms = getTuple();
  MDTuple *GlobalVariables = getTuple();
  MDTuple *ImportedEntities = getTuple();
  uint64_t DWOId = 0x10000000c0ffee;
  auto *N = DICompileUnit::getDistinct(
      Context, SourceLanguage, File, Producer, IsOptimized, Flags,
      RuntimeVersion, SplitDebugFilename, EmissionKind, EnumTypes,
      RetainedTypes, Subprograms, GlobalVariables, ImportedEntities, DWOId);

  EXPECT_EQ(dwarf::DW_TAG_compile_unit, N->getTag());
  EXPECT_EQ(SourceLanguage, N->getSourceLanguage());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Producer, N->getProducer());
  EXPECT_EQ(IsOptimized, N->isOptimized());
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(RuntimeVersion, N->getRuntimeVersion());
  EXPECT_EQ(SplitDebugFilename, N->getSplitDebugFilename());
  EXPECT_EQ(EmissionKind, N->getEmissionKind());
  EXPECT_EQ(EnumTypes, N->getEnumTypes().get());
  EXPECT_EQ(RetainedTypes, N->getRetainedTypes().get());
  EXPECT_EQ(Subprograms, N->getSubprograms().get());
  EXPECT_EQ(GlobalVariables, N->getGlobalVariables().get());
  EXPECT_EQ(ImportedEntities, N->getImportedEntities().get());
  EXPECT_EQ(DWOId, N->getDWOId());

  TempDICompileUnit Temp = N->clone();
  EXPECT_EQ(dwarf::DW_TAG_compile_unit, Temp->getTag());
  EXPECT_EQ(SourceLanguage, Temp->getSourceLanguage());
  EXPECT_EQ(File, Temp->getFile());
  EXPECT_EQ(Producer, Temp->getProducer());
  EXPECT_EQ(IsOptimized, Temp->isOptimized());
  EXPECT_EQ(Flags, Temp->getFlags());
  EXPECT_EQ(RuntimeVersion, Temp->getRuntimeVersion());
  EXPECT_EQ(SplitDebugFilename, Temp->getSplitDebugFilename());
  EXPECT_EQ(EmissionKind, Temp->getEmissionKind());
  EXPECT_EQ(EnumTypes, Temp->getEnumTypes().get());
  EXPECT_EQ(RetainedTypes, Temp->getRetainedTypes().get());
  EXPECT_EQ(Subprograms, Temp->getSubprograms().get());
  EXPECT_EQ(GlobalVariables, Temp->getGlobalVariables().get());
  EXPECT_EQ(ImportedEntities, Temp->getImportedEntities().get());
  EXPECT_EQ(DWOId, Temp->getDWOId());

  auto *TempAddress = Temp.get();
  auto *Clone = MDNode::replaceWithPermanent(std::move(Temp));
  EXPECT_TRUE(Clone->isDistinct());
  EXPECT_EQ(TempAddress, Clone);
}

TEST_F(DICompileUnitTest, replaceArrays) {
  unsigned SourceLanguage = 1;
  DIFile *File = getFile();
  StringRef Producer = "some producer";
  bool IsOptimized = false;
  StringRef Flags = "flag after flag";
  unsigned RuntimeVersion = 2;
  StringRef SplitDebugFilename = "another/file";
  unsigned EmissionKind = 3;
  MDTuple *EnumTypes = MDTuple::getDistinct(Context, None);
  MDTuple *RetainedTypes = MDTuple::getDistinct(Context, None);
  MDTuple *ImportedEntities = MDTuple::getDistinct(Context, None);
  uint64_t DWOId = 0xc0ffee;
  auto *N = DICompileUnit::getDistinct(
      Context, SourceLanguage, File, Producer, IsOptimized, Flags,
      RuntimeVersion, SplitDebugFilename, EmissionKind, EnumTypes,
      RetainedTypes, nullptr, nullptr, ImportedEntities, DWOId);

  auto *Subprograms = MDTuple::getDistinct(Context, None);
  EXPECT_EQ(nullptr, N->getSubprograms().get());
  N->replaceSubprograms(Subprograms);
  EXPECT_EQ(Subprograms, N->getSubprograms().get());
  N->replaceSubprograms(nullptr);
  EXPECT_EQ(nullptr, N->getSubprograms().get());

  auto *GlobalVariables = MDTuple::getDistinct(Context, None);
  EXPECT_EQ(nullptr, N->getGlobalVariables().get());
  N->replaceGlobalVariables(GlobalVariables);
  EXPECT_EQ(GlobalVariables, N->getGlobalVariables().get());
  N->replaceGlobalVariables(nullptr);
  EXPECT_EQ(nullptr, N->getGlobalVariables().get());
}

typedef MetadataTest DISubprogramTest;

TEST_F(DISubprogramTest, get) {
  DIScopeRef Scope = getCompositeType();
  StringRef Name = "name";
  StringRef LinkageName = "linkage";
  DIFile *File = getFile();
  unsigned Line = 2;
  DISubroutineType *Type = getSubroutineType();
  bool IsLocalToUnit = false;
  bool IsDefinition = true;
  unsigned ScopeLine = 3;
  DITypeRef ContainingType = getCompositeType();
  unsigned Virtuality = 4;
  unsigned VirtualIndex = 5;
  unsigned Flags = 6;
  bool IsOptimized = false;
  MDTuple *TemplateParams = getTuple();
  DISubprogram *Declaration = getSubprogram();
  MDTuple *Variables = getTuple();

  auto *N = DISubprogram::get(
      Context, Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit,
      IsDefinition, ScopeLine, ContainingType, Virtuality, VirtualIndex, Flags,
      IsOptimized, TemplateParams, Declaration, Variables);

  EXPECT_EQ(dwarf::DW_TAG_subprogram, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(LinkageName, N->getLinkageName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(IsLocalToUnit, N->isLocalToUnit());
  EXPECT_EQ(IsDefinition, N->isDefinition());
  EXPECT_EQ(ScopeLine, N->getScopeLine());
  EXPECT_EQ(ContainingType, N->getContainingType());
  EXPECT_EQ(Virtuality, N->getVirtuality());
  EXPECT_EQ(VirtualIndex, N->getVirtualIndex());
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(IsOptimized, N->isOptimized());
  EXPECT_EQ(TemplateParams, N->getTemplateParams().get());
  EXPECT_EQ(Declaration, N->getDeclaration());
  EXPECT_EQ(Variables, N->getVariables().get());
  EXPECT_EQ(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, TemplateParams,
                                 Declaration, Variables));

  EXPECT_NE(N, DISubprogram::get(Context, getCompositeType(), Name, LinkageName,
                                 File, Line, Type, IsLocalToUnit, IsDefinition,
                                 ScopeLine, ContainingType, Virtuality,
                                 VirtualIndex, Flags, IsOptimized,
                                 TemplateParams, Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, "other", LinkageName, File,
                                 Line, Type, IsLocalToUnit, IsDefinition,
                                 ScopeLine, ContainingType, Virtuality,
                                 VirtualIndex, Flags, IsOptimized,
                                 TemplateParams, Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, "other", File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, getFile(),
                                 Line, Type, IsLocalToUnit, IsDefinition,
                                 ScopeLine, ContainingType, Virtuality,
                                 VirtualIndex, Flags, IsOptimized,
                                 TemplateParams, Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File,
                                 Line + 1, Type, IsLocalToUnit, IsDefinition,
                                 ScopeLine, ContainingType, Virtuality,
                                 VirtualIndex, Flags, IsOptimized,
                                 TemplateParams, Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 getSubroutineType(), IsLocalToUnit,
                                 IsDefinition, ScopeLine, ContainingType,
                                 Virtuality, VirtualIndex, Flags, IsOptimized,
                                 TemplateParams, Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, !IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, !IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition,
                                 ScopeLine + 1, ContainingType, Virtuality,
                                 VirtualIndex, Flags, IsOptimized,
                                 TemplateParams, Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 getCompositeType(), Virtuality, VirtualIndex,
                                 Flags, IsOptimized, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality + 1, VirtualIndex,
                                 Flags, IsOptimized, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex + 1,
                                 Flags, IsOptimized, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 ~Flags, IsOptimized, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, !IsOptimized, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N,
            DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                              Type, IsLocalToUnit, IsDefinition, ScopeLine,
                              ContainingType, Virtuality, VirtualIndex, Flags,
                              IsOptimized, getTuple(), Declaration, Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, TemplateParams,
                                 getSubprogram(), Variables));
  EXPECT_NE(N, DISubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, TemplateParams,
                                 Declaration, getTuple()));

  TempDISubprogram Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DILexicalBlockTest;

TEST_F(DILexicalBlockTest, get) {
  DILocalScope *Scope = getSubprogram();
  DIFile *File = getFile();
  unsigned Line = 5;
  unsigned Column = 8;

  auto *N = DILexicalBlock::get(Context, Scope, File, Line, Column);

  EXPECT_EQ(dwarf::DW_TAG_lexical_block, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Column, N->getColumn());
  EXPECT_EQ(N, DILexicalBlock::get(Context, Scope, File, Line, Column));

  EXPECT_NE(N,
            DILexicalBlock::get(Context, getSubprogram(), File, Line, Column));
  EXPECT_NE(N, DILexicalBlock::get(Context, Scope, getFile(), Line, Column));
  EXPECT_NE(N, DILexicalBlock::get(Context, Scope, File, Line + 1, Column));
  EXPECT_NE(N, DILexicalBlock::get(Context, Scope, File, Line, Column + 1));

  TempDILexicalBlock Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DILexicalBlockTest, Overflow) {
  DISubprogram *SP = getSubprogram();
  DIFile *F = getFile();
  {
    auto *LB = DILexicalBlock::get(Context, SP, F, 2, 7);
    EXPECT_EQ(2u, LB->getLine());
    EXPECT_EQ(7u, LB->getColumn());
  }
  unsigned U16 = 1u << 16;
  {
    auto *LB = DILexicalBlock::get(Context, SP, F, UINT32_MAX, U16 - 1);
    EXPECT_EQ(UINT32_MAX, LB->getLine());
    EXPECT_EQ(U16 - 1, LB->getColumn());
  }
  {
    auto *LB = DILexicalBlock::get(Context, SP, F, UINT32_MAX, U16);
    EXPECT_EQ(UINT32_MAX, LB->getLine());
    EXPECT_EQ(0u, LB->getColumn());
  }
  {
    auto *LB = DILexicalBlock::get(Context, SP, F, UINT32_MAX, U16 + 1);
    EXPECT_EQ(UINT32_MAX, LB->getLine());
    EXPECT_EQ(0u, LB->getColumn());
  }
}

typedef MetadataTest DILexicalBlockFileTest;

TEST_F(DILexicalBlockFileTest, get) {
  DILocalScope *Scope = getSubprogram();
  DIFile *File = getFile();
  unsigned Discriminator = 5;

  auto *N = DILexicalBlockFile::get(Context, Scope, File, Discriminator);

  EXPECT_EQ(dwarf::DW_TAG_lexical_block, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Discriminator, N->getDiscriminator());
  EXPECT_EQ(N, DILexicalBlockFile::get(Context, Scope, File, Discriminator));

  EXPECT_NE(N, DILexicalBlockFile::get(Context, getSubprogram(), File,
                                       Discriminator));
  EXPECT_NE(N,
            DILexicalBlockFile::get(Context, Scope, getFile(), Discriminator));
  EXPECT_NE(N,
            DILexicalBlockFile::get(Context, Scope, File, Discriminator + 1));

  TempDILexicalBlockFile Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DINamespaceTest;

TEST_F(DINamespaceTest, get) {
  DIScope *Scope = getFile();
  DIFile *File = getFile();
  StringRef Name = "namespace";
  unsigned Line = 5;

  auto *N = DINamespace::get(Context, Scope, File, Name, Line);

  EXPECT_EQ(dwarf::DW_TAG_namespace, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(N, DINamespace::get(Context, Scope, File, Name, Line));

  EXPECT_NE(N, DINamespace::get(Context, getFile(), File, Name, Line));
  EXPECT_NE(N, DINamespace::get(Context, Scope, getFile(), Name, Line));
  EXPECT_NE(N, DINamespace::get(Context, Scope, File, "other", Line));
  EXPECT_NE(N, DINamespace::get(Context, Scope, File, Name, Line + 1));

  TempDINamespace Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DIModuleTest;

TEST_F(DIModuleTest, get) {
  DIScope *Scope = getFile();
  StringRef Name = "module";
  StringRef ConfigMacro = "-DNDEBUG";
  StringRef Includes = "-I.";
  StringRef Sysroot = "/";

  auto *N = DIModule::get(Context, Scope, Name, ConfigMacro, Includes, Sysroot);

  EXPECT_EQ(dwarf::DW_TAG_module, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(ConfigMacro, N->getConfigurationMacros());
  EXPECT_EQ(Includes, N->getIncludePath());
  EXPECT_EQ(Sysroot, N->getISysRoot());
  EXPECT_EQ(N, DIModule::get(Context, Scope, Name,
                             ConfigMacro, Includes, Sysroot));
  EXPECT_NE(N, DIModule::get(Context, getFile(), Name,
                             ConfigMacro, Includes, Sysroot));
  EXPECT_NE(N, DIModule::get(Context, Scope, "other",
                             ConfigMacro, Includes, Sysroot));
  EXPECT_NE(N, DIModule::get(Context, Scope, Name,
                             "other", Includes, Sysroot));
  EXPECT_NE(N, DIModule::get(Context, Scope, Name,
                             ConfigMacro, "other", Sysroot));
  EXPECT_NE(N, DIModule::get(Context, Scope, Name,
                             ConfigMacro, Includes, "other"));

  TempDIModule Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DITemplateTypeParameterTest;

TEST_F(DITemplateTypeParameterTest, get) {
  StringRef Name = "template";
  DITypeRef Type = getBasicType("basic");

  auto *N = DITemplateTypeParameter::get(Context, Name, Type);

  EXPECT_EQ(dwarf::DW_TAG_template_type_parameter, N->getTag());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(N, DITemplateTypeParameter::get(Context, Name, Type));

  EXPECT_NE(N, DITemplateTypeParameter::get(Context, "other", Type));
  EXPECT_NE(N,
            DITemplateTypeParameter::get(Context, Name, getBasicType("other")));

  TempDITemplateTypeParameter Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DITemplateValueParameterTest;

TEST_F(DITemplateValueParameterTest, get) {
  unsigned Tag = dwarf::DW_TAG_template_value_parameter;
  StringRef Name = "template";
  DITypeRef Type = getBasicType("basic");
  Metadata *Value = getConstantAsMetadata();

  auto *N = DITemplateValueParameter::get(Context, Tag, Name, Type, Value);
  EXPECT_EQ(Tag, N->getTag());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(Value, N->getValue());
  EXPECT_EQ(N, DITemplateValueParameter::get(Context, Tag, Name, Type, Value));

  EXPECT_NE(N, DITemplateValueParameter::get(
                   Context, dwarf::DW_TAG_GNU_template_template_param, Name,
                   Type, Value));
  EXPECT_NE(N,
            DITemplateValueParameter::get(Context, Tag, "other", Type, Value));
  EXPECT_NE(N, DITemplateValueParameter::get(Context, Tag, Name,
                                             getBasicType("other"), Value));
  EXPECT_NE(N, DITemplateValueParameter::get(Context, Tag, Name, Type,
                                             getConstantAsMetadata()));

  TempDITemplateValueParameter Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DIGlobalVariableTest;

TEST_F(DIGlobalVariableTest, get) {
  DIScope *Scope = getSubprogram();
  StringRef Name = "name";
  StringRef LinkageName = "linkage";
  DIFile *File = getFile();
  unsigned Line = 5;
  DITypeRef Type = getDerivedType();
  bool IsLocalToUnit = false;
  bool IsDefinition = true;
  Constant *Variable = getConstant();
  DIDerivedType *StaticDataMemberDeclaration =
      cast<DIDerivedType>(getDerivedType());

  auto *N = DIGlobalVariable::get(Context, Scope, Name, LinkageName, File, Line,
                                  Type, IsLocalToUnit, IsDefinition, Variable,
                                  StaticDataMemberDeclaration);
  EXPECT_EQ(dwarf::DW_TAG_variable, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(LinkageName, N->getLinkageName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(IsLocalToUnit, N->isLocalToUnit());
  EXPECT_EQ(IsDefinition, N->isDefinition());
  EXPECT_EQ(Variable, N->getVariable());
  EXPECT_EQ(StaticDataMemberDeclaration, N->getStaticDataMemberDeclaration());
  EXPECT_EQ(N, DIGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, IsLocalToUnit, IsDefinition,
                                     Variable, StaticDataMemberDeclaration));

  EXPECT_NE(N,
            DIGlobalVariable::get(Context, getSubprogram(), Name, LinkageName,
                                  File, Line, Type, IsLocalToUnit, IsDefinition,
                                  Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, "other", LinkageName, File,
                                     Line, Type, IsLocalToUnit, IsDefinition,
                                     Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, Name, "other", File, Line,
                                     Type, IsLocalToUnit, IsDefinition,
                                     Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N,
            DIGlobalVariable::get(Context, Scope, Name, LinkageName, getFile(),
                                  Line, Type, IsLocalToUnit, IsDefinition,
                                  Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N,
            DIGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                  Line + 1, Type, IsLocalToUnit, IsDefinition,
                                  Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N,
            DIGlobalVariable::get(Context, Scope, Name, LinkageName, File, Line,
                                  getDerivedType(), IsLocalToUnit, IsDefinition,
                                  Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, !IsLocalToUnit, IsDefinition,
                                     Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N, DIGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, IsLocalToUnit, !IsDefinition,
                                     Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N,
            DIGlobalVariable::get(Context, Scope, Name, LinkageName, File, Line,
                                  Type, IsLocalToUnit, IsDefinition,
                                  getConstant(), StaticDataMemberDeclaration));
  EXPECT_NE(N,
            DIGlobalVariable::get(Context, Scope, Name, LinkageName, File, Line,
                                  Type, IsLocalToUnit, IsDefinition, Variable,
                                  cast<DIDerivedType>(getDerivedType())));

  TempDIGlobalVariable Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DILocalVariableTest;

TEST_F(DILocalVariableTest, get) {
  DILocalScope *Scope = getSubprogram();
  StringRef Name = "name";
  DIFile *File = getFile();
  unsigned Line = 5;
  DITypeRef Type = getDerivedType();
  unsigned Arg = 6;
  unsigned Flags = 7;

  auto *N =
      DILocalVariable::get(Context, Scope, Name, File, Line, Type, Arg, Flags);
  EXPECT_TRUE(N->isParameter());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(Arg, N->getArg());
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(N, DILocalVariable::get(Context, Scope, Name, File, Line, Type, Arg,
                                    Flags));

  EXPECT_FALSE(
      DILocalVariable::get(Context, Scope, Name, File, Line, Type, 0, Flags)
          ->isParameter());
  EXPECT_NE(N, DILocalVariable::get(Context, getSubprogram(), Name, File, Line,
                                    Type, Arg, Flags));
  EXPECT_NE(N, DILocalVariable::get(Context, Scope, "other", File, Line, Type,
                                    Arg, Flags));
  EXPECT_NE(N, DILocalVariable::get(Context, Scope, Name, getFile(), Line, Type,
                                    Arg, Flags));
  EXPECT_NE(N, DILocalVariable::get(Context, Scope, Name, File, Line + 1, Type,
                                    Arg, Flags));
  EXPECT_NE(N, DILocalVariable::get(Context, Scope, Name, File, Line,
                                    getDerivedType(), Arg, Flags));
  EXPECT_NE(N, DILocalVariable::get(Context, Scope, Name, File, Line, Type,
                                    Arg + 1, Flags));
  EXPECT_NE(N, DILocalVariable::get(Context, Scope, Name, File, Line, Type, Arg,
                                    ~Flags));

  TempDILocalVariable Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DILocalVariableTest, getArg256) {
  EXPECT_EQ(255u, DILocalVariable::get(Context, getSubprogram(), "", getFile(),
                                       0, nullptr, 255, 0)
                      ->getArg());
  EXPECT_EQ(256u, DILocalVariable::get(Context, getSubprogram(), "", getFile(),
                                       0, nullptr, 256, 0)
                      ->getArg());
  EXPECT_EQ(257u, DILocalVariable::get(Context, getSubprogram(), "", getFile(),
                                       0, nullptr, 257, 0)
                      ->getArg());
  unsigned Max = UINT16_MAX;
  EXPECT_EQ(Max, DILocalVariable::get(Context, getSubprogram(), "", getFile(),
                                      0, nullptr, Max, 0)
                     ->getArg());
}

typedef MetadataTest DIExpressionTest;

TEST_F(DIExpressionTest, get) {
  uint64_t Elements[] = {2, 6, 9, 78, 0};
  auto *N = DIExpression::get(Context, Elements);
  EXPECT_EQ(makeArrayRef(Elements), N->getElements());
  EXPECT_EQ(N, DIExpression::get(Context, Elements));

  EXPECT_EQ(5u, N->getNumElements());
  EXPECT_EQ(2u, N->getElement(0));
  EXPECT_EQ(6u, N->getElement(1));
  EXPECT_EQ(9u, N->getElement(2));
  EXPECT_EQ(78u, N->getElement(3));
  EXPECT_EQ(0u, N->getElement(4));

  TempDIExpression Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(DIExpressionTest, isValid) {
#define EXPECT_VALID(...)                                                      \
  do {                                                                         \
    uint64_t Elements[] = {__VA_ARGS__};                                       \
    EXPECT_TRUE(DIExpression::get(Context, Elements)->isValid());              \
  } while (false)
#define EXPECT_INVALID(...)                                                    \
  do {                                                                         \
    uint64_t Elements[] = {__VA_ARGS__};                                       \
    EXPECT_FALSE(DIExpression::get(Context, Elements)->isValid());             \
  } while (false)

  // Empty expression should be valid.
  EXPECT_TRUE(DIExpression::get(Context, None));

  // Valid constructions.
  EXPECT_VALID(dwarf::DW_OP_plus, 6);
  EXPECT_VALID(dwarf::DW_OP_deref);
  EXPECT_VALID(dwarf::DW_OP_bit_piece, 3, 7);
  EXPECT_VALID(dwarf::DW_OP_plus, 6, dwarf::DW_OP_deref);
  EXPECT_VALID(dwarf::DW_OP_deref, dwarf::DW_OP_plus, 6);
  EXPECT_VALID(dwarf::DW_OP_deref, dwarf::DW_OP_bit_piece, 3, 7);
  EXPECT_VALID(dwarf::DW_OP_deref, dwarf::DW_OP_plus, 6, dwarf::DW_OP_bit_piece, 3, 7);

  // Invalid constructions.
  EXPECT_INVALID(~0u);
  EXPECT_INVALID(dwarf::DW_OP_plus);
  EXPECT_INVALID(dwarf::DW_OP_bit_piece);
  EXPECT_INVALID(dwarf::DW_OP_bit_piece, 3);
  EXPECT_INVALID(dwarf::DW_OP_bit_piece, 3, 7, dwarf::DW_OP_plus, 3);
  EXPECT_INVALID(dwarf::DW_OP_bit_piece, 3, 7, dwarf::DW_OP_deref);

#undef EXPECT_VALID
#undef EXPECT_INVALID
}

typedef MetadataTest DIObjCPropertyTest;

TEST_F(DIObjCPropertyTest, get) {
  StringRef Name = "name";
  DIFile *File = getFile();
  unsigned Line = 5;
  StringRef GetterName = "getter";
  StringRef SetterName = "setter";
  unsigned Attributes = 7;
  DITypeRef Type = getBasicType("basic");

  auto *N = DIObjCProperty::get(Context, Name, File, Line, GetterName,
                                SetterName, Attributes, Type);

  EXPECT_EQ(dwarf::DW_TAG_APPLE_property, N->getTag());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(GetterName, N->getGetterName());
  EXPECT_EQ(SetterName, N->getSetterName());
  EXPECT_EQ(Attributes, N->getAttributes());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(N, DIObjCProperty::get(Context, Name, File, Line, GetterName,
                                   SetterName, Attributes, Type));

  EXPECT_NE(N, DIObjCProperty::get(Context, "other", File, Line, GetterName,
                                   SetterName, Attributes, Type));
  EXPECT_NE(N, DIObjCProperty::get(Context, Name, getFile(), Line, GetterName,
                                   SetterName, Attributes, Type));
  EXPECT_NE(N, DIObjCProperty::get(Context, Name, File, Line + 1, GetterName,
                                   SetterName, Attributes, Type));
  EXPECT_NE(N, DIObjCProperty::get(Context, Name, File, Line, "other",
                                   SetterName, Attributes, Type));
  EXPECT_NE(N, DIObjCProperty::get(Context, Name, File, Line, GetterName,
                                   "other", Attributes, Type));
  EXPECT_NE(N, DIObjCProperty::get(Context, Name, File, Line, GetterName,
                                   SetterName, Attributes + 1, Type));
  EXPECT_NE(N, DIObjCProperty::get(Context, Name, File, Line, GetterName,
                                   SetterName, Attributes,
                                   getBasicType("other")));

  TempDIObjCProperty Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest DIImportedEntityTest;

TEST_F(DIImportedEntityTest, get) {
  unsigned Tag = dwarf::DW_TAG_imported_module;
  DIScope *Scope = getSubprogram();
  DINodeRef Entity = getCompositeType();
  unsigned Line = 5;
  StringRef Name = "name";

  auto *N = DIImportedEntity::get(Context, Tag, Scope, Entity, Line, Name);

  EXPECT_EQ(Tag, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Entity, N->getEntity());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(N, DIImportedEntity::get(Context, Tag, Scope, Entity, Line, Name));

  EXPECT_NE(N,
            DIImportedEntity::get(Context, dwarf::DW_TAG_imported_declaration,
                                  Scope, Entity, Line, Name));
  EXPECT_NE(N, DIImportedEntity::get(Context, Tag, getSubprogram(), Entity,
                                     Line, Name));
  EXPECT_NE(N, DIImportedEntity::get(Context, Tag, Scope, getCompositeType(),
                                     Line, Name));
  EXPECT_NE(N,
            DIImportedEntity::get(Context, Tag, Scope, Entity, Line + 1, Name));
  EXPECT_NE(N,
            DIImportedEntity::get(Context, Tag, Scope, Entity, Line, "other"));

  TempDIImportedEntity Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest MetadataAsValueTest;

TEST_F(MetadataAsValueTest, MDNode) {
  MDNode *N = MDNode::get(Context, None);
  auto *V = MetadataAsValue::get(Context, N);
  EXPECT_TRUE(V->getType()->isMetadataTy());
  EXPECT_EQ(N, V->getMetadata());

  auto *V2 = MetadataAsValue::get(Context, N);
  EXPECT_EQ(V, V2);
}

TEST_F(MetadataAsValueTest, MDNodeMDNode) {
  MDNode *N = MDNode::get(Context, None);
  Metadata *Ops[] = {N};
  MDNode *N2 = MDNode::get(Context, Ops);
  auto *V = MetadataAsValue::get(Context, N2);
  EXPECT_TRUE(V->getType()->isMetadataTy());
  EXPECT_EQ(N2, V->getMetadata());

  auto *V2 = MetadataAsValue::get(Context, N2);
  EXPECT_EQ(V, V2);

  auto *V3 = MetadataAsValue::get(Context, N);
  EXPECT_TRUE(V3->getType()->isMetadataTy());
  EXPECT_NE(V, V3);
  EXPECT_EQ(N, V3->getMetadata());
}

TEST_F(MetadataAsValueTest, MDNodeConstant) {
  auto *C = ConstantInt::getTrue(Context);
  auto *MD = ConstantAsMetadata::get(C);
  Metadata *Ops[] = {MD};
  auto *N = MDNode::get(Context, Ops);

  auto *V = MetadataAsValue::get(Context, MD);
  EXPECT_TRUE(V->getType()->isMetadataTy());
  EXPECT_EQ(MD, V->getMetadata());

  auto *V2 = MetadataAsValue::get(Context, N);
  EXPECT_EQ(MD, V2->getMetadata());
  EXPECT_EQ(V, V2);
}

typedef MetadataTest ValueAsMetadataTest;

TEST_F(ValueAsMetadataTest, UpdatesOnRAUW) {
  Type *Ty = Type::getInt1PtrTy(Context);
  std::unique_ptr<GlobalVariable> GV0(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  auto *MD = ValueAsMetadata::get(GV0.get());
  EXPECT_TRUE(MD->getValue() == GV0.get());
  ASSERT_TRUE(GV0->use_empty());

  std::unique_ptr<GlobalVariable> GV1(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  GV0->replaceAllUsesWith(GV1.get());
  EXPECT_TRUE(MD->getValue() == GV1.get());
}

TEST_F(ValueAsMetadataTest, CollidingDoubleUpdates) {
  // Create a constant.
  ConstantAsMetadata *CI = ConstantAsMetadata::get(
      ConstantInt::get(getGlobalContext(), APInt(8, 0)));

  // Create a temporary to prevent nodes from resolving.
  auto Temp = MDTuple::getTemporary(Context, None);

  // When the first operand of N1 gets reset to nullptr, it'll collide with N2.
  Metadata *Ops1[] = {CI, CI, Temp.get()};
  Metadata *Ops2[] = {nullptr, CI, Temp.get()};

  auto *N1 = MDTuple::get(Context, Ops1);
  auto *N2 = MDTuple::get(Context, Ops2);
  ASSERT_NE(N1, N2);

  // Tell metadata that the constant is getting deleted.
  //
  // After this, N1 will be invalid, so don't touch it.
  ValueAsMetadata::handleDeletion(CI->getValue());
  EXPECT_EQ(nullptr, N2->getOperand(0));
  EXPECT_EQ(nullptr, N2->getOperand(1));
  EXPECT_EQ(Temp.get(), N2->getOperand(2));

  // Clean up Temp for teardown.
  Temp->replaceAllUsesWith(nullptr);
}

typedef MetadataTest TrackingMDRefTest;

TEST_F(TrackingMDRefTest, UpdatesOnRAUW) {
  Type *Ty = Type::getInt1PtrTy(Context);
  std::unique_ptr<GlobalVariable> GV0(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  TypedTrackingMDRef<ValueAsMetadata> MD(ValueAsMetadata::get(GV0.get()));
  EXPECT_TRUE(MD->getValue() == GV0.get());
  ASSERT_TRUE(GV0->use_empty());

  std::unique_ptr<GlobalVariable> GV1(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  GV0->replaceAllUsesWith(GV1.get());
  EXPECT_TRUE(MD->getValue() == GV1.get());

  // Reset it, so we don't inadvertently test deletion.
  MD.reset();
}

TEST_F(TrackingMDRefTest, UpdatesOnDeletion) {
  Type *Ty = Type::getInt1PtrTy(Context);
  std::unique_ptr<GlobalVariable> GV(
      new GlobalVariable(Ty, false, GlobalValue::ExternalLinkage));
  TypedTrackingMDRef<ValueAsMetadata> MD(ValueAsMetadata::get(GV.get()));
  EXPECT_TRUE(MD->getValue() == GV.get());
  ASSERT_TRUE(GV->use_empty());

  GV.reset();
  EXPECT_TRUE(!MD);
}

TEST(NamedMDNodeTest, Search) {
  LLVMContext Context;
  ConstantAsMetadata *C =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Context), 1));
  ConstantAsMetadata *C2 =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Context), 2));

  Metadata *const V = C;
  Metadata *const V2 = C2;
  MDNode *n = MDNode::get(Context, V);
  MDNode *n2 = MDNode::get(Context, V2);

  Module M("MyModule", Context);
  const char *Name = "llvm.NMD1";
  NamedMDNode *NMD = M.getOrInsertNamedMetadata(Name);
  NMD->addOperand(n);
  NMD->addOperand(n2);

  std::string Str;
  raw_string_ostream oss(Str);
  NMD->print(oss);
  EXPECT_STREQ("!llvm.NMD1 = !{!0, !1}\n",
               oss.str().c_str());
}

typedef MetadataTest FunctionAttachmentTest;
TEST_F(FunctionAttachmentTest, setMetadata) {
  Function *F = getFunction("foo");
  ASSERT_FALSE(F->hasMetadata());
  EXPECT_EQ(nullptr, F->getMetadata(LLVMContext::MD_dbg));
  EXPECT_EQ(nullptr, F->getMetadata("dbg"));
  EXPECT_EQ(nullptr, F->getMetadata("other"));

  DISubprogram *SP1 = getSubprogram();
  DISubprogram *SP2 = getSubprogram();
  ASSERT_NE(SP1, SP2);

  F->setMetadata("dbg", SP1);
  EXPECT_TRUE(F->hasMetadata());
  EXPECT_EQ(SP1, F->getMetadata(LLVMContext::MD_dbg));
  EXPECT_EQ(SP1, F->getMetadata("dbg"));
  EXPECT_EQ(nullptr, F->getMetadata("other"));

  F->setMetadata(LLVMContext::MD_dbg, SP2);
  EXPECT_TRUE(F->hasMetadata());
  EXPECT_EQ(SP2, F->getMetadata(LLVMContext::MD_dbg));
  EXPECT_EQ(SP2, F->getMetadata("dbg"));
  EXPECT_EQ(nullptr, F->getMetadata("other"));

  F->setMetadata("dbg", nullptr);
  EXPECT_FALSE(F->hasMetadata());
  EXPECT_EQ(nullptr, F->getMetadata(LLVMContext::MD_dbg));
  EXPECT_EQ(nullptr, F->getMetadata("dbg"));
  EXPECT_EQ(nullptr, F->getMetadata("other"));

  MDTuple *T1 = getTuple();
  MDTuple *T2 = getTuple();
  ASSERT_NE(T1, T2);

  F->setMetadata("other1", T1);
  F->setMetadata("other2", T2);
  EXPECT_TRUE(F->hasMetadata());
  EXPECT_EQ(T1, F->getMetadata("other1"));
  EXPECT_EQ(T2, F->getMetadata("other2"));
  EXPECT_EQ(nullptr, F->getMetadata("dbg"));

  F->setMetadata("other1", T2);
  F->setMetadata("other2", T1);
  EXPECT_EQ(T2, F->getMetadata("other1"));
  EXPECT_EQ(T1, F->getMetadata("other2"));

  F->setMetadata("other1", nullptr);
  F->setMetadata("other2", nullptr);
  EXPECT_FALSE(F->hasMetadata());
  EXPECT_EQ(nullptr, F->getMetadata("other1"));
  EXPECT_EQ(nullptr, F->getMetadata("other2"));
}

TEST_F(FunctionAttachmentTest, getAll) {
  Function *F = getFunction("foo");

  MDTuple *T1 = getTuple();
  MDTuple *T2 = getTuple();
  MDTuple *P = getTuple();
  DISubprogram *SP = getSubprogram();

  F->setMetadata("other1", T2);
  F->setMetadata(LLVMContext::MD_dbg, SP);
  F->setMetadata("other2", T1);
  F->setMetadata(LLVMContext::MD_prof, P);
  F->setMetadata("other2", T2);
  F->setMetadata("other1", T1);

  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  F->getAllMetadata(MDs);
  ASSERT_EQ(4u, MDs.size());
  EXPECT_EQ(LLVMContext::MD_dbg, MDs[0].first);
  EXPECT_EQ(LLVMContext::MD_prof, MDs[1].first);
  EXPECT_EQ(Context.getMDKindID("other1"), MDs[2].first);
  EXPECT_EQ(Context.getMDKindID("other2"), MDs[3].first);
  EXPECT_EQ(SP, MDs[0].second);
  EXPECT_EQ(P, MDs[1].second);
  EXPECT_EQ(T1, MDs[2].second);
  EXPECT_EQ(T2, MDs[3].second);
}

TEST_F(FunctionAttachmentTest, dropUnknownMetadata) {
  Function *F = getFunction("foo");

  MDTuple *T1 = getTuple();
  MDTuple *T2 = getTuple();
  MDTuple *P = getTuple();
  DISubprogram *SP = getSubprogram();

  F->setMetadata("other1", T1);
  F->setMetadata(LLVMContext::MD_dbg, SP);
  F->setMetadata("other2", T2);
  F->setMetadata(LLVMContext::MD_prof, P);

  unsigned Known[] = {Context.getMDKindID("other2"), LLVMContext::MD_prof};
  F->dropUnknownMetadata(Known);

  EXPECT_EQ(T2, F->getMetadata("other2"));
  EXPECT_EQ(P, F->getMetadata(LLVMContext::MD_prof));
  EXPECT_EQ(nullptr, F->getMetadata("other1"));
  EXPECT_EQ(nullptr, F->getMetadata(LLVMContext::MD_dbg));

  F->setMetadata("other2", nullptr);
  F->setMetadata(LLVMContext::MD_prof, nullptr);
  EXPECT_FALSE(F->hasMetadata());
}

TEST_F(FunctionAttachmentTest, Verifier) {
  Function *F = getFunction("foo");
  F->setMetadata("attach", getTuple());

  // Confirm this has no body.
  ASSERT_TRUE(F->empty());

  // Functions without a body cannot have metadata attachments (they also can't
  // be verified directly, so check that the module fails to verify).
  EXPECT_TRUE(verifyModule(*F->getParent()));

  // Functions with a body can.
  (void)new UnreachableInst(Context, BasicBlock::Create(Context, "bb", F));
  EXPECT_FALSE(verifyModule(*F->getParent()));
  EXPECT_FALSE(verifyFunction(*F));
}

TEST_F(FunctionAttachmentTest, EntryCount) {
  Function *F = getFunction("foo");
  EXPECT_FALSE(F->getEntryCount().hasValue());
  F->setEntryCount(12304);
  EXPECT_TRUE(F->getEntryCount().hasValue());
  EXPECT_EQ(12304u, *F->getEntryCount());
}

TEST_F(FunctionAttachmentTest, SubprogramAttachment) {
  Function *F = getFunction("foo");
  DISubprogram *SP = getSubprogram();
  F->setSubprogram(SP);

  // Note that the static_cast confirms that F->getSubprogram() actually
  // returns an DISubprogram.
  EXPECT_EQ(SP, static_cast<DISubprogram *>(F->getSubprogram()));
  EXPECT_EQ(SP, F->getMetadata("dbg"));
  EXPECT_EQ(SP, F->getMetadata(LLVMContext::MD_dbg));
}

}
