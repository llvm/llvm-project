#ifndef RECOVER_FROM_LLVM_IR
#define RECOVER_FROM_LLVM_IR

#include <stack>
#include <string>
#include <unordered_map>
#include <utility> // For std::pair

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/InstrTypes.h"

// See http://llvm.org/docs/ProgrammersManual.html#isa
#include "llvm/Support/Casting.h" // For dyn_cast

#define EXCHANGE(a, b)                                                         \
  a ^= b;                                                                      \
  b ^= a;                                                                      \
  a ^= b;

#ifndef MAXLEN_STR
#define MAXLEN_STR 8192
#endif

#include "Misc.h"

// #define DEBUG_TYPE LV_NAME
// #define LLVM_DEBUG DEBUG

// static const std::string STR_REMAINDER_VF = "n.mod.vf";

using namespace llvm;

namespace {

// Normally used to return the variable name without suffix e.g. ".034"
std::string rStripStringAfterChar(std::string str, char ch) {
  std::size_t pos = str.find(ch);

  return str.substr(0, pos);
}

/* Important Note:
 *   If the val is an LLVM variable, it will return something like
 *      "%[llvm_var_name]".
 *   If val is a constant it returns normally the value of the
 *      constant.
 *
 *  I consider a rather big defficiency of Value::getName() NOT to return
 *      (itself or a different method, created by the key LLVM people)
 *      the auto-generated number like %0, if the Value is created without an
 *      explicit name.
 *
 *   Important: I noticed that for different Instruction the result of print()
 *      can be somewhat different, like:
 *      - i32 %0
 *      - %1 = bitcast ...
 */
std::string getLLVMValueName(Value *val) {
  /* Somewhat important: it is possible that, if the API
    changes a bit the name will NOT be printed
    here anymore */
  std::string printStr;
  raw_string_ostream OS(printStr);

  // bci->printAsOperand(OS, true); // Does NOT write anything (false neither)

  // See http://llvm.org/docs/doxygen/html/Value_8h_source.html#l00202
  /* Note: IsForDebug false can print:
         - the SAME as true or
         - the complete instruction, not just the value */
  val->print(OS, /*IsForDebug*/ true);
  LLVM_DEBUG(dbgs() << "getLLVMValueName(): printStr = " << printStr << "\n");

  std::string strValName;
  std::string strValName2;

  if (llvm::dyn_cast<Constant>(val) != nullptr) {
    LLVM_DEBUG(dbgs() << "getLLVMValueName(): val is Constant\n");

    // See http://llvm.org/docs/doxygen/html/classllvm_1_1Constant.html
    // sscanf(printStr.c_str(), "%s %s", strValName2, strValName);
    strValName2 = stringScanf(printStr, (char *)"%s ");
    strValName =
        stringScanf(printStr.substr(strValName2.length()), (char *)"%s ");

    /* Normally printStr is of form "type_ct val_ct".
     *  But we can also have something like
     *   @dataT = common local_unnamed_addr global [128 x [150 x half]]
     *                                                          zeroinitializer
     */
    if (strValName2[0] == '@')
      strValName = strValName2.substr(1);
  } else {
    std::size_t posPercent = printStr.find('%');
    LLVM_DEBUG(dbgs() << "getLLVMValueName(): posPercent = " << posPercent
                      << "\n");

    if (posPercent == std::string::npos) {
      // This is NOT a variable Value - probably just a constant
      return ""; // std::to_string("");
    }

    // sscanf(printStr.substr(posPercent).c_str(), "%s ", strValName);
    strValName = stringScanf(printStr.substr(posPercent), (char *)"%s ");
    // sscanf(valTypeAndName.c_str(), "%s %s", strValName, strValName);
  }

  std::string res = strValName;
  LLVM_DEBUG(dbgs() << "getLLVMValueName(): res = " << res << "\n");

  return res;
}

// Used by getAllMetadata() (and getExpr())
bool ranGetAllMetadata;
// DenseMap<Value *, std::string> varNameMap;
// Map with <name of Value, name of source var represented>
std::unordered_map<std::string, std::string> varNameMap;
//
void getAllMetadata(Function *F) {
  ranGetAllMetadata = true;

  LLVM_DEBUG(dbgs() << "Entered getAllMetadata()\n");

  // Some info about metadata:
  //              http://llvm.org/docs/SourceLevelDebugging.html#llvm-dbg-value

  // Inspired from
  // weaponshot.wordpress.com/2012/05/06/extract-all-the-metadata-nodes-in-llvm/
  for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      /* Get the Metadata declared in the llvm intrinsic functions
          such as llvm.dbg.declare() */
      if (CallInst *CI = dyn_cast<CallInst>(I)) {
        if (Function *F = CI->getCalledFunction()) {
          // We look at the llvm.dbg.value metadata which associates Value
          //   (LLVM IR values) with names in the original program
          if (F->getName().starts_with("llvm.dbg.value")) {
          // if (F->getName().startswith("llvm.dbg"))
            LLVM_DEBUG(dbgs() << "getAllMetadata(): CI = " << *CI << "\n");

            /* It seems that the association between LLVM IR
                Value and names in the original source program
                is always like this:
                    - opnd 0 contains the Value,
                    - opnd 1 is always a (useless?) 0,
                    - opnd 2 contains the DILocalVariable,
            */
            // Error: <<no known conversion for argument 1 from
            //   ‘const llvm::Value*’ to ‘const llvm::Metadata*’>>:
            //   DILocalVariable *srcVar = llvm::dyn_cast_or_null<
            //                               DILocalVariable>(I->getOperand(2));
            // Error: <<no known conversion for argument 1 from
            //  ‘const llvm::Value*’ to ‘const llvm::Metadata*’>>:
            // MDNode *srcVar =
            // llvm::dyn_cast_or_null<MDNode>(I->getOperand(2));
            /* See llvm.org/docs/doxygen/html/classllvm_1_1MetadataAsValue.html
            (see maybe llvm.org/docs/doxygen/html/namespacellvm_1_1mdconst.html:
                "Now that Value and Metadata are in separate hierarchies" */
            MetadataAsValue *srcVarMDV =
                llvm::dyn_cast_or_null<MetadataAsValue>(I->getOperand(2));

            // Value *val = I->getOperand(0);
            MetadataAsValue *val =
                llvm::dyn_cast_or_null<MetadataAsValue>(I->getOperand(0));
            assert(val != nullptr);

            if (srcVarMDV != nullptr) {
              // See http://llvm.org/docs/doxygen/html/classllvm_1_1MDNode.html
              // MDNode *srcVar = llvm::dyn_cast_or_null<MDNode>(
              //                                      srcVarMDV->getMetadata());

              // See
              // llvm.org/docs/doxygen/html/classllvm_1_1DILocalVariable.html
              //    and llvm.org/docs/doxygen/html/classllvm_1_1DIVariable.html
              DILocalVariable *srcVar = llvm::dyn_cast_or_null<DILocalVariable>(
                  srcVarMDV->getMetadata());

              assert(srcVar != nullptr);

              // Gives compiler-error:
              //            const MDOperand srcVarOpnd0 = srcVar->getOperand(0);
              // const MDOperand *srcVarOpnd0 = & (srcVar->getOperand(0));

              std::string valueName = getLLVMValueName(val);
              if (valueName.size() == 0) {
                /* We can have metadata which has for 1st
                    operand a constant e.g. 0.
                  For ex
                   call void @llvm.dbg.value(metadata i32 0, i64 0,
                                           metadata !32, metadata !21), !dbg !33
                */
                continue;
              }

              // varNameMap[valTypeAndName] = (srcVar->getName()).str();
              varNameMap[valueName] = (srcVar->getName()).str();

              // See
              // llvm.org/docs/doxygen/html/classllvm_1_1DILocalVariable.html
              LLVM_DEBUG(dbgs() << "getAllMetadata(): val = " << *val << "\n");
              LLVM_DEBUG(dbgs() << "    val = " << val << "\n");
              LLVM_DEBUG(dbgs() << "    val->getValueName() = "
                                << val->getValueName() << "\n");
              LLVM_DEBUG(dbgs()
                         << "    val->getName() = " << val->getName() << "\n");
              LLVM_DEBUG(dbgs() << "    srcVar = " << *srcVar << "\n");
              // LLVM_DEBUG(dbgs() << "    srcVar->getOperand(0) = "
              LLVM_DEBUG(dbgs() << "    srcVarName = "
                                << varNameMap[valueName] /* srcVar->getName() */
                                << "\n");
            }
          }
        }
      }
    }
  }
} // end getAllMetadata()

std::string printCTypeFromLLVMType(Type *aType, LLVMContext *aContext) {
  std::string res;

  // See http://llvm.org/doxygen/classllvm_1_1Type.html
  if (aType == Type::getInt16Ty(*aContext))
    res = "short";
  else if (aType == Type::getInt32Ty(*aContext)) // Builder.getInt32Ty())
    res = "int";
  else if (aType == Type::getHalfTy(*aContext))
    res = "half";
  else
    assert(0 && "printCTypeFromLLVMType(): Type NOT supported");

  return res;
}

// TODO: probably we will need to treat struct/record,
//    union/variants
Type *getElementTypeOfDerivedType(Type *valType) {
  int sizeofElem;

  LLVM_DEBUG(dbgs() << "getElementTypeOfDerivedType(): valType = " << *valType
                    << "\n");

  // Helps for vector type.
  // So it does NOT help for pointer type, as it is the case for val (normally).
  Type *scalarType = valType->getScalarType();
  LLVM_DEBUG(dbgs() << "getElementTypeOfDerivedType(): scalarType = "
                    << *scalarType << "\n");

  sizeofElem = scalarType->getScalarSizeInBits() / 8;
  LLVM_DEBUG(dbgs() << "getElementTypeOfDerivedType(): sizeof(scalarType) = "
                    << sizeofElem << "\n");
  if (sizeofElem != 0)
    return scalarType;

  /*
  // Does not help: both return 0...
  LLVM_DEBUG(dbgs() << "GetSize(): bitsizeof(type of val) = "
                  //<< valType->getPrimitiveSizeInBits() / 8 << "\n");
                    << valType->getScalarSizeInBits() << "\n");
  */
  ArrayType *arrType = llvm::dyn_cast<ArrayType>(valType);

  if (arrType != nullptr) {
    Type *elemArrType = arrType->getElementType();
    sizeofElem = elemArrType->getScalarSizeInBits() / 8;
    LLVM_DEBUG(
        dbgs()
        << "getElementTypeOfDerivedType(): (arrType != nullptr): elemArrType = "
        << *elemArrType << "\n");
    LLVM_DEBUG(
        dbgs()
        << "getElementTypeOfDerivedType(): (arrType != nullptr): sizeofElem = "
        << sizeofElem << "\n");

    if (sizeofElem == 0) {
      return getElementTypeOfDerivedType(elemArrType);
    } else {
      return elemArrType;
    }
  }

  /*
  // MEGA-TODO: Now LLVM has opaque pointer types
  //              - see https://llvm.org/docs/OpaquePointers.html
  // See http://llvm.org/docs/doxygen/html/classllvm_1_1PointerType.html
  PointerType *ptrType = llvm::dyn_cast<PointerType>(valType);
  if (ptrType != nullptr) {
    Type *elemPtrType = ptrType->getElementType();

    sizeofElem = elemPtrType->getScalarSizeInBits() / 8;
    LLVM_DEBUG(dbgs() << "getElementTypeOfDerivedType(): elemPtrType = "
                      << *elemPtrType << "\n");
    LLVM_DEBUG(dbgs() << "getElementTypeOfDerivedType(): sizeof(elemPtrType) = "
                      << sizeofElem << "\n");

    if (sizeofElem == 0) {
      return getElementTypeOfDerivedType(elemPtrType);
    }
    else
      return elemPtrType;
  }
  */

  return nullptr;
}

bool testEquivalence(Instruction *it, PHINode *phi) {
  Value *op0 = nullptr;

  LLVM_DEBUG(dbgs() << "Entered testEquivalence(): it = " << *it
                    << ", phi = " << *phi << "\n");

  if (phi == it)
    return true;

  if (it->getNumOperands() > 0) {
    op0 = it->getOperand(0);
  }

  switch (it->getOpcode()) {
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::Trunc:
    // case Instruction::ShuffleVector:
    // case Instruction::InsertElement:
    // case Instruction::PHI:
    // case Instruction::ExtractElement:
    // res = "";
    break;

    /* case Instruction::GetElementPtr:
      break; */

  default:
    return false;
    // assert(0 && "testEquivalence(): we do not deal with these cases");
  }

  /*
  // Important-TODO: need to do this for the case we have an access like
  //    B[j + 1][0]
  switch (it->getOpcode()) {
    case Instruction::Add:
      res += " + ";
      break;
  }
  */

  return testEquivalence((Instruction *)op0, phi);
}

inline bool isGlobalArray(GetElementPtrInst *GEPPtr) {
  return llvm::dyn_cast<GlobalValue>(GEPPtr->getOperand(0)) != nullptr;
}
//
inline int getIndexFirstOpndFromGEPInst(GetElementPtrInst *GEPPtr) {
  int startIndex;

  if (isGlobalArray(GEPPtr)) {
    /* Following also
      http://llvm.org/docs/GetElementPtr.html#why-is-the-extra-0-index-required
       we see that for global arrays, the 1st index
       in GEP is redundant - it has value 0 invariably,
       so we skip it.
    */
    startIndex = 2;
  } else {
    startIndex = 1;
  }

  return startIndex;
}
//
inline Value *getFirstIndexOpndFromGEPInst(GetElementPtrInst *GEPPtr) {
  int startIndex = getIndexFirstOpndFromGEPInst(GEPPtr);

  Value *res = GEPPtr->getOperand(startIndex);

  return res;
}

// We check we have a correctly paranthesized expression
bool checkCorrectParanthesis(std::string res, int *indexTop) {
  std::stack<int> stack;
  int numOpenParans = 0;

  for (std::size_t i = 0; i < res.length(); i++) {
    if (res[i] == '(') {
      numOpenParans++;
      stack.push(i);
    }
    if (res[i] == ')') {
      numOpenParans--;
      if (indexTop != nullptr)
        *indexTop = stack.top();

      // assert(numOpenParans >= 0 && "Invalid arithmetic expression!");
      if (numOpenParans < 0)
        return false;

      stack.pop();
    }
  }

  return (numOpenParans == 0);
}

/*
Currently this function ONLY does this: it gets rid of duplicated spaces.

// Important-TODO: get rid of unnecessary parantheses
  - for this normally I have to parse expr before and pretty-print it
     intelligently.
To do algebraic simplification is more complex. See Muchnick's book,
    - value numbering, etc.

To do Constant folding (Constant-Expression Evaluation),
     although both these methods are heavy, we could use them:
  We could try to use CIL's partial evaluation module, but:
    - it doesn't work with C++
  We can't use sympy, which can parse expressions (parse_expr) and
       simplify them (method sympy.simplify.cse_main.cse) because:
    - see e.g.
       https://github.com/sympy/sympy/blob/master/sympy/parsing/sympy_parser.py
    - it doesn't handle pointers, etc - but we can extend it
    - ...
*/
std::string canonicalizeExpression(std::string aStr,
                                   bool removeOuterParens = false) {
  std::string res = aStr;

  LLVM_DEBUG(dbgs() << "Entered canonicalizeExpression(aStr = " << aStr
                    << ")\n");

  // We make all double spaces single-spaces.
  for (;;) {
    // From http://www.cplusplus.com/reference/string/string/find/
    std::size_t pos = res.find("  ");

    if (pos == std::string::npos) {
      break;
    } else {
      // std::cout << "first 'needle' found at position: " << pos << "\n";
      // From http://www.cplusplus.com/reference/string/string/erase/
      res.erase(pos, 1);
    }
  }
  /*
  std::cout << "canonicalizeExpression(): returning aStr = "
            << res << "\n";
  */

  if (removeOuterParens) {
    int indexTop = -1;

    bool correct = checkCorrectParanthesis(res, &indexTop);
    // assert(correct == true && "Invalid arithmetic expression!");

    LLVM_DEBUG(dbgs() << "canonicalizeExpression(): indexTop = " << indexTop
                      << "\n");
    if (indexTop == 0) {
      // If the last paranthesis (not necessarily the last char) is matched by
      //    the 1st paranthesis
      // assert(res.front() == '(' && res.back() == ')');
      if (res.front() == '(' && res.back() == ')') {
        LLVM_DEBUG(
            dbgs() << "canonicalizeExpression(): removing useless outer ()\n");

        res = res.substr(1, res.size() - 2);
      }
    }
  }

  if (res != aStr)
    return canonicalizeExpression(res);
  return res;
}

inline void printInfo(Instruction *it, std::string str0, std::string str1,
                      std::string iGetNameData, Value *op0, Value *op1) {
  LLVM_DEBUG(dbgs() << "printInfo(): it = " << *it << "\n");
  LLVM_DEBUG(dbgs() << "printInfo(): it ptr = " << it << "\n");
  LLVM_DEBUG(dbgs() << "    (printInfo(): it->getOpcodeName() = "
                    << it->getOpcodeName() << ")\n");
  LLVM_DEBUG(dbgs() << "    (printInfo(): it->getOpcode() = " << it->getOpcode()
                    << ")\n");
  LLVM_DEBUG(dbgs() << "    (printInfo(): it->getName() = " << iGetNameData
                    << ")\n");
  LLVM_DEBUG(dbgs() << "    (printInfo(): str0 = " << str0 << ")\n");
  LLVM_DEBUG(dbgs() << "    (printInfo(): str1 = " << str1 << ")\n");

  if (op0 == nullptr) {
    LLVM_DEBUG(dbgs() << "    (printInfo(): op0 = nullptr\n");
  } else {
    LLVM_DEBUG(dbgs() << "    (printInfo(): op0 = " << *op0 << ")\n");
  }

  if (op1 == nullptr) {
    LLVM_DEBUG(dbgs() << "    (printInfo(): op1 = nullptr\n");
  } else {
    LLVM_DEBUG(dbgs() << "    (printInfo(): op1 = " << *op1 << ")\n");
  }
}

std::string getPredicateString(int pred) {
  std::string res;

  // See all values of enum Predicate at
  //                         https://llvm.org/doxygen/classllvm_1_1CmpInst.html
  // Note: FCMP_O* is for ordered (neither operand can be a QNAN),
  //   FCMP_O* is for unordered (either can be QNAN) -
  //   see https://llvm.org/docs/LangRef.html#fcmp-instruction
  if (pred == CmpInst::ICMP_SGT || pred == CmpInst::ICMP_UGT ||
      pred == CmpInst::FCMP_OGT || pred == CmpInst::FCMP_UGT)
    res = " > ";
  else if (pred == CmpInst::ICMP_SGE || pred == CmpInst::ICMP_UGE ||
           pred == CmpInst::FCMP_OGE || pred == CmpInst::FCMP_UGE)
    res = " >= ";
  else if (pred == CmpInst::ICMP_SLT || pred == CmpInst::ICMP_ULT ||
           pred == CmpInst::FCMP_OLT || pred == CmpInst::FCMP_ULT)
    res = " < ";
  else if (pred == CmpInst::ICMP_SLE || pred == CmpInst::ICMP_ULE ||
           pred == CmpInst::FCMP_OLE || pred == CmpInst::FCMP_ULE)
    res = " <= ";
  else if (pred == CmpInst::ICMP_EQ || pred == CmpInst::FCMP_OEQ ||
           pred == CmpInst::FCMP_UEQ)
    res = " == ";
  else if (pred == CmpInst::ICMP_NE || pred == CmpInst::FCMP_ONE ||
           pred == CmpInst::FCMP_UNE)
    res = " != ";

  return res;
}

/* Alex:
 *  - we get a C expression
 *    by walking on the use-def-chains (more exactly the only reaching
 *    definition for the SSA it instruction) in order to get the most complete
 *    definition for the it instruction.
 *
 *  - doing some sort of partial evaluation

  Note: SCEV also pretty prints - display expressions related to tripcounts
    (zext i16 (-1 + %N) to i32)
       (see code below:
        BackedgeTakenCount->dump();
        ExitCount->dump(); )
  See, more exactly,
             http://llvm.org/docs/doxygen/html/ScalarEvolution_8cpp_source.html
    void SCEV::print(raw_ostream &OS) const {}

  Important Note: We use ((int *)&x) instead of &x because the & for an array
         (global at least) is a pointer to array and this affects/reflects on
         the pointer arithmetic.
    Concrete example on ARM 32 on zedboard.arh.pub.ro:
   /home/alarm/OpincaaLLVM/opincaa_standalone_app/35_MatMul/SIZE_256/STDout_003a
        Before 1st write: &A = 405912
        Before 1st write: &A + 20 = 3027352
        Before 1st write: &A + 131072 = 405912
        Before 1st write: ((char *)&A) + 131072 = 536984
      when running on ARM (32 bits processor) it is possible that &A + x == &A
         (where x is e.g. 131072) (probably because of overflow or because the
         VM did not map memory there or...)
   So, again, we need to use when doing arithmetic instead of &A --> (int *)(&A)
        or (short/char *)(&A) .
   Note: [TODO CHECK WELL]: It seems for pointer type we print just the
         var e.g. A without &A.
 */
int getExprExitCount_StepCt_int = -100100100;
bool usePaddingForNestedLoops_more = false;
bool getExprVarSpecial = false;
// Very Important: call also cacheExpr.clear(); when making
//                                                getExprVarSpecial = true;
// IIRC used to distinguish between Value LLVM IR objects with the same name
// (by adding the pointer address to it, thus resulting name is e.g.
//   row__0x1299b68)
// bool getExprForTripCount = false;
bool getExprForDMATransfer = true;
// std::unordered_map<Instruction *, std::string> cacheExpr;
std::unordered_map<Value *, std::string> cacheExpr;
Value *basePtrGetExprIt; // The base pointer (GetElementPtr, 1st operand)
int getExprGEPCount = 0;
Value *getExprGEP;
// Important-TODO: make getExpr(Value *it) and check if it is instruction or not
//
std::string getExpr(Value *aVal) {
  std::string str0("");
  std::string str1("");

  std::string strCopy;

  // ConstantExpr: bool isSelectConstantExpr = false;
  const ConstantExpr *CE;

  std::string res;

  Value *op0 = nullptr;
  Value *op1 = nullptr;

  const std::string STR_VEC_IND = "vec.ind";
  const std::string STR_STEP_ADD = "step.add";
  /* Note that if I recall correctly, the var names ending in splatinsert are
   automatically generated */
  const std::string STR_BROADCAST_SPLATINSERT = "broadcast.splatinsert";
  const std::string STR_SPLATINSERT = ".splatinsert";
  const std::string STR_BROADCAST_SPLAT = "broadcast.splat";
  const std::string STR_SPLAT = ".splat";
  //
  const std::string STR_INDUCTION = "induction";
  const std::string STR_UNDEF = "undef";
  const std::string STR_INDEX = "index";
  const std::string STR_INDEX_NEXT = "index.next";

  const std::string INVALID_VALUE_CACHEEXPR = "\\@@INVALID_STR@@";

  // See http://www.cplusplus.com/reference/unordered_map/unordered_map/find/
  // std::unordered_map<Instruction *, std::string>::const_iterator got =
  std::unordered_map<Value *, std::string>::const_iterator got =
      // cacheExpr.find(it);
      cacheExpr.find(aVal);

  // const char *iGetNameData = it->getName().data();
  // std::string iGetNameData(it->getName().data());
  std::string iGetNameData(aVal->getName().data());

  res.clear();

  if (aVal == nullptr) {
    LLVM_DEBUG(dbgs() << "Entered getExpr(): aVal = nullptr.\n");
    return std::string("");
  } else {
    LLVM_DEBUG(dbgs() << "Entered getExpr(): *aVal = " << *aVal << ".\n");
  }

  Instruction *it = llvm::dyn_cast<Instruction>(aVal);
  if (it == nullptr) {
    LLVM_DEBUG(dbgs() << "getExpr(): it = nullptr.\n");

    res = "";

    /*
    it = (Instruction *)aVal;
    LLVM_DEBUG(dbgs() << "getExpr(): After static typecast, *it = "
                      << *it << ".\n");
    */

    /* Global var (values, not arrays) in LLVM language are already pointers to
        the global address space. This is why we need to use & for them.
     We check that *it is a GlobalValue like:
        @colsK = common local_unnamed_addr global i32 0, align 4
    // See http://llvm.org/docs/doxygen/html/classllvm_1_1GlobalValue.html
     //   (also http://llvm.org/docs/LangRef.html#global-variables)
    */
    // if (GlobalValue *gv = llvm::dyn_cast<GlobalValue>(it))
    if (llvm::dyn_cast<GlobalValue>(aVal) != nullptr) {
      if (usePaddingForNestedLoops_more == true)
        res = "(";
      else
        res = "((int *)&";

      if (getExprVarSpecial) {
        // res += "<VAR*SPECIAL>";
      }

      res += iGetNameData;

      if (getExprVarSpecial) {
        res += stringPrintf(const_cast<char *>("__%p"), (void *)it);
        // res += "<VAR*SPECIAL*END>";
      }

      res += ")";
      if (basePtrGetExprIt == nullptr)
        basePtrGetExprIt = aVal;

      goto GetExpr_end;
    }

    /* See llvm.org/docs/doxygen/html/Core_8h_source.html#l00100 and
       http://llvm.org/docs/doxygen/html/Instruction_8cpp_source.html#l00194
       for all supported opcodes.
       In fact, we can have more valid opcodes than these
        See http://llvm.org/docs/doxygen/html/Core_8h_source.html#l00100
            - the enums with typedef enum LLVMOpcode - e.g., LLVMAdd, etc
                seem to be related to values of Instruction::getOpcode().
                 I think Instruction:Add == LLVMAdd + InstructionVal
                                                      (use gdb to see exactly);
               note also that getOpcode() returns getValueID() - InstructionVal.
        http://llvm.org/docs/doxygen/html/Value_8h_source.html
            see enum ValueTy - better see
                          http://llvm.org/test-doxygen/api/Value_8h_source.html,
            since the Value.h source file uses TableGen macros inside.
    */
    LLVM_DEBUG(dbgs() << "getExpr(): Special case\n");
    const Constant *C = llvm::dyn_cast<Constant>(aVal);

    LLVM_DEBUG(dbgs() << "getExpr(): C = " << C << "\n");

    if (C != nullptr) {
      LLVM_DEBUG(dbgs() << "  getExpr(): aVal is Constant.\n");
      // res += "Constant-->";

      if (const ConstantInt *CI = llvm::dyn_cast<ConstantInt>(C)) {
        LLVM_DEBUG(dbgs() << "  getExpr(): CI->getValue() = " << CI->getValue()
                          << ".\n");
        res += std::to_string(CI->getValue().getSExtValue());
      } else if (const ConstantDataVector *CDV =
                     llvm::dyn_cast<ConstantDataVector>(C)) {
        LLVM_DEBUG(dbgs() << "  getExpr(): CDV->getSplatValue() = "
                          << CDV->getSplatValue() << ".\n");
        res += std::to_string(
            ((ConstantInt *)CDV->getSplatValue())->getSExtValue());
      }
      else
      /*
      // Maybe useful in the future, but little likely:
      if (const ConstantDataArray *CA =
      llvm::dyn_cast<ConstantDataArray>(C)) { LLVM_DEBUG(dbgs() << "
      getExpr(): It is ConstantDataArray.\n");
      }
      if (const ConstantArray *CA = llvm::dyn_cast<ConstantArray>(C)) {
          LLVM_DEBUG(dbgs() << "  getExpr(): It is ConstantArray.\n");
      }
      */

      /* Inspired from
        http://llvm.org/docs/doxygen/html/AsmWriter_8cpp_source.html#l01304,
          method WriteConstantInternal() .
      */
      if (CE = llvm::dyn_cast<ConstantExpr>(C)) {
        LLVM_DEBUG(dbgs() << "  getExpr(): It is ConstantExpr.\n");
        LLVM_DEBUG(dbgs() << "  getExpr(): CE->getNumOperands() = "
                          << CE->getNumOperands() << "\n");

        if (CE->getNumOperands() > 0) {
          op0 = CE->getOperand(0);
          str0 = op0->getName().data();

          if (CE->getOpcode() == Instruction::GetElementPtr) {
            res += "(int *)&(" + str0;
          } else {
            res += "(" + getExpr(op0) + " )";
          }
          // small-TODO: Replace the " )" with ")"
        }

        // From http://llvm.org/test-doxygen/api/Constants_8cpp_source.html
        // res += CE->getOpcodeName();
        switch (CE->getOpcode()) {
        // small-TODO: This code is similar to the one for the switch above
        //   - maybe we should reuse code although it will make things more
        //   complicated...
        case Instruction::Add:
          res += " + ";
          break;
        case Instruction::Sub:
          res += " - ";
          break;
        case Instruction::Mul:
          res += " * ";
          break;
        case Instruction::UDiv:
        case Instruction::SDiv:
          res += " / ";
          break;
        case Instruction::SRem:
        case Instruction::URem:
          res += " % ";
          break;
        case Instruction::Shl:
          res += " << ";
          break;
        case Instruction::LShr:
          res += " >> ";
          break;
        case Instruction::AShr:
          res += " >> ";
          break;
        case Instruction::ICmp:
        case Instruction::FCmp: {
          // Check type of cmp
          /*
          int pred = CE->getPredicate();
          res += getPredicateString(pred);
          */
          res += std::string(CE->getOpcodeName()); // MEGA-TODO: check if correct

          break;
        }
        case Instruction::ZExt:
        case Instruction::SExt:
          // res += " ext "; // Note: this is unary operator
          break;
        case Instruction::Trunc:
          // res += " trunc ";
          break;
        case Instruction::Select: {
          res += " ? ";

          // To pretty-print the 3rd operand, below
          /*
          // ConstantExpr: isSelectConstantExpr = true;
          LLVM_DEBUG(dbgs()
                    << "getExpr(): setting isSelectConstantExpr = true\n");
          */
          break;
        }

        case Instruction::PtrToInt:
        case Instruction::IntToPtr: {
          break;
        }
        case Instruction::GetElementPtr: {
          // res += " [Unsupported_C_CtExpr_operator = GEP]";

          int numOpnds = CE->getNumOperands();
          int startIndex = 2; // getIndexFirstOpndFromGEPInst(GEPPtr);

          for (int i = startIndex; i < numOpnds; i++) {
            res += "[";

            Value *op_i = CE->getOperand(i);
            res += getExpr(op_i);

            res += "]";
          }
          break;
        }
        default:
          res += " [Unsupported_C_CtExpr_operator]";
          break;
        }

        if (CE->getOpcode() != Instruction::GetElementPtr &&
            CE->getNumOperands() > 1) {
          op1 = CE->getOperand(1);
          str1 = op1->getName().data();

          res += getExpr(op1);
        }

        if (CE->getOpcode() == Instruction::Select &&
            CE->getNumOperands() > 2) {
          res += " : ";

          op1 = CE->getOperand(2);
          str1 = op1->getName().data();

          res += getExpr(op1);
        }

        if (CE->getNumOperands() == 0) // MEGA-TODO: takeout this silly check
          res += " ";

        if (CE->getOpcode() == Instruction::GetElementPtr) {
          res += ")";
        }
      } // END if (CE = llvm::dyn_cast<ConstantExpr>(C))
      else {
        res += " [Unsupported_Constant]";
        /*
        // Compiler error: <<error: cannot use typeid with -fno-rtti>>
        res += typeid(C).name() + " ";
        res += typeid(aVal).name() + " ";
        */

        res += " ";
      }

      // break;
      // goto GetExpr_end;
    } else {
      // res += " [Unsupported_C_operator] [Constant_C_is_nullptr]";
      res += iGetNameData;

      // goto GetExpr_end;
    }

    goto GetExpr_end;
  } else {
    LLVM_DEBUG(dbgs() << "getExpr(): *it = " << *it << ".\n");
  }

  /* Note: It is possible that the names have a suffix when we have 2+
     vars starting with the same name - this happens when more
     vector.body BBs are created (more loops are vectorized).
     For this, we use strncmp(), not strcmp(). */

  if (it->getNumOperands() > 0) {
    op0 = it->getOperand(0);
    str0 = op0->getName().data();
    if (it->getNumOperands() > 1) {
      op1 = it->getOperand(1);
      str1 = op1->getName().data();
    }
  }

  /*
   * Note: It points to an Instruction (or just a Value).
     getOperand() returns type Value.
   * From http://llvm.org/docs/doxygen/html/classllvm_1_1Value.html
       << StringRef   getName () const
         Return a constant reference to the value's name. >>
   */

  /*
  LLVM_DEBUG(dbgs() << "getExpr(): getExprForTripCount = "
                    << getExprForTripCount << "\n");
  */
  printInfo(it, str0, str1, iGetNameData, op0, op1);

  if (got == cacheExpr.end()) {
    // cacheExpr.insert(it);

    /* We insert an empty string res, just to keep track we visited this
     * node and we update the entry with the correct value at the end of
     * the function. */
    cacheExpr[it] = INVALID_VALUE_CACHEEXPR; // res;
  } else {
    if (cacheExpr[it] != INVALID_VALUE_CACHEEXPR) {
      // This case can be quite easily reached if the expression it has
      //   several times as constituent atoms the same expression.
      res = got->second;
      LLVM_DEBUG(
          dbgs()
          << "getExpr(): We already visited this node so we stop here.\n");
      goto GetExpr_end;
    }
    else
    /* We have already cached something for this node,
     * either an INVALID_VALUE_CACHEEXPR or a valid value we can return
     * directly. */
    if (it->getOpcode() == Instruction::PHI) {
      /* If we visited this phi we do NOT revisit it since it can easily
       * result in infinite cycles... It's not very fundamented,
       * but it's OK :) */
      // We should keep the unstripped name, although it is possible that if
      //   we visited the variable node before it might be already stripped.

      if (str0.empty()) {
        std::string exprOp0 = getExpr(op0);
        LLVM_DEBUG(dbgs() << "getExpr(): Checking PHI's exprOp0 = " << exprOp0
                          << " (should be a constant).\n");

        // MEGA-TODO: Test well, also regressive tests
        if (iGetNameData.empty()) {
          if (exprOp0.size() > 4)
            res = exprOp0;
        }

        // assert(exprOp0 == "0");
      } else if (str1.empty()) {
        std::string exprOp1 = getExpr(op1);
        LLVM_DEBUG(dbgs() << "getExpr(): Checking PHI's exprOp1 = " << exprOp1
                          << " (should be a constant).\n");

        // MEGA-TODO: test well, also regressive tests
        if (iGetNameData.empty()) {
          if (exprOp1.size() > 4)
            res = exprOp1;
        }

        // assert(exprOp1 == "0");
      } else {
        LLVM_DEBUG(dbgs() << "getExpr(): Setting res to empty string.\n");

        res = rStripStringAfterChar(iGetNameData, '.');
        if (getExprVarSpecial) {
          res += stringPrintf(const_cast<char *>("__%p"), (void *)it);

          // res += "<VAR*SPECIAL*END>";
        }

        goto GetExpr_end;
      }

      LLVM_DEBUG(
          dbgs()
          << "getExpr(): We visited part of this PHI node "
             "so we approximate it... This should be avoided if possible.\n");

      if (getExprVarSpecial) {
        // res += "<VAR*SPECIAL>";
      }

      LLVM_DEBUG(dbgs() << "getExpr(): res = " << res << "\n");
      // res = rStripStringAfterChar(iGetNameData, '.');
      strCopy.assign(iGetNameData);
      strCopy = rStripStringAfterChar(strCopy, '.');
      res += strCopy;
      LLVM_DEBUG(dbgs() << "getExpr(): after, res = " << res << "\n");

      if (getExprVarSpecial) {
        res += stringPrintf(const_cast<char *>("__%p"), (void *)it);

        // res += "<VAR*SPECIAL*END>";
      }

      goto GetExpr_end;
    }
  } // END else if (got == cacheExpr.end())

  /* // NOT_TREAT_NMODVF
  // When computing trip count, I don't want it to be multiple of VF,
  //    but I want the original expression.
  // Note: n.mod.vf is a name given by the program below (this module) in
  //   getOrCreateVectorTripCount().
  // It is possible that the names to have a suffix since the names
  //   exist, since a different vector.body was created before.
  if (startsWith(iGetNameData, STR_REMAINDER_VF)) {
    LLVM_DEBUG(dbgs() << "getExpr(): NOT following remainder var "
                      << iGetNameData << ".\n");

    // A simple hack, since I already have the - operator and am lazy to
    //   get rid of it:
    res = "0";

    goto GetExpr_end;
  }
  */

  if (startsWith(iGetNameData, STR_INDUCTION) &&
      (it->getOpcode() == Instruction::Add)) {
    LLVM_DEBUG(dbgs() << "getExpr(): NOT following induction var "
                      << iGetNameData << ".\n");
    res = getExpr(it->getOperand(0));

    /* Indeed, induction is a vector of consecutive indices - let's call it
     a vector index.
     Very Important: To understand things better, we distinguish:
        - the scalar index, indexLLVM_LV, or LV's index (and index.next)
        - the vector index, vec.ind, used for loading from array (well,
        sortof scalar, but...) */

    /*
    // We do NOT process this:
    res += " + ";
    // TODO: check that op1 == <VF x i...><0, 1, ..., VF-1>
    res += "indexLLVM_LV";
    */
    goto GetExpr_end;
  }

  if ((it->getOpcode() == Instruction::PHI) &&
      startsWith(iGetNameData, STR_INDEX) &&
      startsWith(std::string(it->getOperand(1)->getName().data()),
                 STR_INDEX_NEXT)) {
    // TODO Check that op0 is constant 0.
    // Coping with %index = phi i32 [ 0, %vector.ph ],
    //                                           [ %index.next, %vector.body ]
    // LLVM_DEBUG(dbgs() << "getExpr(): NOT following index induction var.\n");
    LLVM_DEBUG(dbgs() << "getExpr(): Treating special case index = "
                         "phi(0, index.next).\n");

    // A simple hack, since I already have the - operator and I am lazy to
    //   get rid of it:
#ifdef AGGREGATED_DMA_TRANSFERS
    // Important note: we include this file from the back end also now
    //         (not only LoopVectorize.cpp)
    if (getExprForDMATransfer)
      res = "0";
    else
      res = "indexLLVM_LV";
#else
    if (getExprForDMATransfer)
      res = "0";
    else
      res = "indexLLVM_LV";
#endif

    goto GetExpr_end;
  }

  // Here we try to solve a recurrence equation with any PHI node related to
  //   the C source variables:
  if ((it->getOpcode() == Instruction::PHI) &&
      startsWith(iGetNameData, STR_VEC_IND) == false &&
      startsWith(iGetNameData, STR_STEP_ADD) == false &&
      startsWith(iGetNameData, STR_INDUCTION) == false) {
    LLVM_DEBUG(
        dbgs()
        << "getExpr(): it is Phi, phi node with no special vector vars...\n");

    assert(it->getNumOperands() > 0);

    // std::string exprOp1 = canonicalizeExpression(getExpr(op1);

    /*
    // Treating the rather unfortunate case that scalar evolution
    //    is "over-intelligent"
    #define STR_INDVAR_WITH_ASSOCIATED_PHI "indvarScEv"
    if (startsWith(iGetNameData, STR_INDVAR_WITH_ASSOCIATED_PHI)) {
      LLVM_DEBUG(dbgs() << "getExpr(): Treating Phi case indvar with "
                           "associated Phis\n");

      BasicBlock *bbIt = it->getParent();

      // TODO: retrieve increment step of Instruction *it
      //   (maybe use the SCEV BackedgeTakenCount of getOrCreateTripCount()).
      // for (auto instr : bbIt)
      // for (BasicBlock::iterator I = bbIt->begin(); isa<PHINode>(I); ++I)
      for (const Instruction &I : *bbIt) {
        if (I.getOpcode() == Instruction::PHI) {
          std::string Iname = I.getName();
          LLVM_DEBUG(dbgs() << "getExpr(): Iname = " << Iname << "\n");

          if (!startsWith(Iname, STR_INDVAR_WITH_ASSOCIATED_PHI)) {
            // MEGA-TODO: check *I has the same PHI labels

            // TODO: retrieve increment step of Instruction *I

            res = getExpr(&I);
            goto GetExpr_end;
            //break;
          }
        }
      }
    }
    */

    // MEGA-TODO: Test well
    if (((Instruction *)op0)->getOpcode() == Instruction::PHI) {
      // MEGA-TODO:
      // && startsWith(exprOp1, STR_UNDEF))
      LLVM_DEBUG(dbgs() << "getExpr(): op0 is Phi --> res = getExpr(op0)\n");

      // NOT good - doesn't work well e.g. for 41_PolybenchCPU_covariance for
      //         the for (j = i; ...) loop: res = getExpr(op0);
      res = rStripStringAfterChar(iGetNameData, '.');
      // Temporary solution: MEGA-TODO: test well

      if (getExprVarSpecial) { // NOT tested, not sure if really required
        res += stringPrintf(const_cast<char *>("__%p"), (void *)it);
        // res += "<VAR*SPECIAL*END>";
      }

      goto GetExpr_end;
    }
    else
    // MEGA-TODO: Test well
    if (it->getNumOperands() >= 2)
      // This is required e.g. for
      //        41_PolybenchCPU_covariance/B/A_With_counter_instrument/5/test.c
      if (((Instruction *)op1)->getOpcode() == Instruction::PHI) {
        // MEGA-TODO: && startsWith(exprOp1, STR_UNDEF)
        LLVM_DEBUG(dbgs() << "getExpr(): op1 is Phi --> res = getExpr(op1)\n");
        res = getExpr(op1);
        goto GetExpr_end;
      }

    std::string exprOp0Orig =
        canonicalizeExpression(getExpr(op0),
                               /* removeOuterParens = */ true);
    LLVM_DEBUG(dbgs() << "    exprOp0Orig = " << exprOp0Orig << "\n");
    if (exprOp0Orig == "0") {
      // if (str0.empty())
      // Note: constants like i64 0 don't have name --> str0 is empty
      // Note: vars like %33 but also constants like i64 0 don't have name -->
      // str0 is empty
      LLVM_DEBUG(
          dbgs() << "  getExpr(): strlen(str0) == 0 --> exchanging operands\n");

      str0.swap(str1);

      Value *tmp = op0;
      op0 = op1;
      op1 = tmp;

      // EXCHANGE(str0, str1);
      // EXCHANGE((int)op0, (int)op1);

      printInfo(it, str0, str1, iGetNameData, op0, op1);
    }

    /*
    // TODO: We should treat PHI in a unified way by putting op0 to be a
      // constant normally and op1 an add/sub Instruction (normally)
      // rather-important-MEGA-TODO: determine if BB Phi-target of op0 dominates
      // BB Phi-target of op1. If not exchange op0<->op1
      // Currently we do a simpler solution since we do not have an
      //      instance of DomTree in getExpr().
      //   This solution is good but it does NOT treat cases like
      //       for (j = i + 1; j < ...; j +...)
      std::string exprOp0 = getExpr(op0);
      std::string exprOp1 = getExpr(op1);
      if (exprOp1[0] >= '0' && exprOp1[0] <= '9') {
        LLVM_DEBUG(dbgs() << "    Exchanging op0 and op1\n");

        // op1 is a numerical constant - we exchange 0 with 1
        Value *tmp = op1;
        op1 = op0;
        op0 = tmp;
      }
    */
    if (str0.empty() == false) {
      // If op0 is not a constant (since the name of op0 is NOT empty)
      // assert(str0 ==(symbolically, after more recovery) iGetNameData + 1);

      LLVM_DEBUG(dbgs() << "    ... *op0 = " << *op0 << "\n");
      LLVM_DEBUG(dbgs() << "    ... *op1 = " << *op1 << "\n");

      LLVM_DEBUG(dbgs() << "    ... Entering getExpr() for op0\n");
      std::string exprOp0 =
          canonicalizeExpression(getExpr(op0),
                                 /* removeOuterParens = */ false);
      LLVM_DEBUG(dbgs() << "    exprOp0 = " << exprOp0 << "\n");

      std::string itVarPlusCt = "(";

      strCopy.assign(iGetNameData);
      strCopy = rStripStringAfterChar(strCopy, '.');
      if (getExprVarSpecial) {
        strCopy += stringPrintf(const_cast<char *>("__%p"), (void *)it);
      }

      // assert(op1 != nullptr);
      Instruction *itOp0 = llvm::dyn_cast<Instruction>(op0);
      assert(itOp0 != nullptr);
      std::string exprItOp0Op0 = getExpr(itOp0->getOperand(0));
      LLVM_DEBUG(dbgs() << "    strCopy = " << strCopy << "\n");
      LLVM_DEBUG(dbgs() << "    exprItOp1Op0 = " << exprItOp0Op0 << "\n");
      assert(exprItOp0Op0 == strCopy);

      // itVarPlusCt = itVarPlusCt + iGetNameData;
      itVarPlusCt = itVarPlusCt + strCopy;
      //
      // itVarPlusCt = itVarPlusCt + " + 1)";
      itVarPlusCt = itVarPlusCt + " + " + getExpr(itOp0->getOperand(1)) + ")";

      LLVM_DEBUG(dbgs() << "    exprOp0 = " << exprOp0 << "\n");
      LLVM_DEBUG(dbgs() << "    itVarPlusCt = " << itVarPlusCt << "\n");

      if (exprOp0 != itVarPlusCt) {
        // Important-TODO: take from the other if case below
        LLVM_DEBUG(dbgs() << "    (SPECIAL) VERY BAD case encountered: "
                          << "Phi node is NOT like x = Phi(x + 1, 0) --> "
                             "return 'main' part of exprOp0 = "
                          << exprOp0 << "\n");

        /* Important-TODO: this case is special (with a ~incomplete solution)
           - to compute a solution to the phi node we normally require more
             intelligent analysis.

        Here we treat associated Phis:
        For example, for test 32_MatAdd we have:
          %conv48.us = phi i32 [ %conv.us, %for.cond3.for.inc12_crit_edge.us ],
                              [ 0, %for.cond3.preheader.us.preheader ]
          %i.047.us = phi i16 [ %inc13.us, %for.cond3.for.inc12_crit_edge.us ],
                              [ 0, %for.cond3.preheader.us.preheader ]

        While the 2nd phi has an easy to find solution (by seeing that
          %inc13.us = add i16 %i.047.us, 1, !dbg !27)
          which means the closed-form solution of Phi is %i.047.us = i,
         for the 1st phi node the situation is Very complicated.
        But we see that:
           %conv.us = sext i16 %inc13.us to i32, !dbg !28
         which makes the Phi expression of %conv48.us the same as
            for %i.047.us .

        Also for SSD:
            %conv48.us = phi(i.047.us + 1, 0)

            getExpr(): it =   %conv327 = phi i32 [ 0, %for.cond2.preheader ],
                                                 [ %conv3, %for.inc44 ]
            getExpr():     op1 =   %conv3 = sext i16 %inc45 to i32, !dbg !41
            getExpr():     updated op1 =   %inc45 = add i16 %counter.026, 1,
                                                                       !dbg !40
        Alhough %conv.327 does NOT appear in the final .ll file, if we look in:
            NEW_v128i16/90_CV/SSD/STDerr_clang_opt_01
        we have a similar case:
          for.cond7.preheader:
                                     ; preds = %for.cond2.preheader, %for.inc44
            %conv327 = phi i32 [ 0, %for.cond2.preheader ],
                               [ %conv3, %for.inc44 ]
            %counter.026 = phi i16 [ 0, %for.cond2.preheader ],
                                   [ %inc45, %for.inc44 ]
        */

        /* MEGA-TODO: think if possible to do better like
         having getExpr return a parse tree where it is clear that a
         node is a var or constant in order to avoid using substr. */
        /*
        LLVM_DEBUG(dbgs()
                       << "      res = " << res << "\n"
                       << "      exprOp0.substr(1, exprOp0.size() - 6) = "
                       << exprOp0.substr(1, exprOp0.size() - 6) << "\n");
        */
        assert(exprOp0.size() >= 6 + 1);

        // Since we are in a SPECIAL ("VERY BAD") case, we make some
        //   checks to "fix" the problem. If it doesn't work we simply
        //   "bail" out
        std::string exprOp0Substr = exprOp0.substr(1, exprOp0.size() - 6);
        LLVM_DEBUG(dbgs() << "    exprOp0.substr(1, exprOp0.size() - 6) = "
                          << exprOp0Substr);
        if (checkCorrectParanthesis(exprOp0Substr, nullptr) &&
            (endsWith(exprOp0Substr, " +") ==
             false) // MEGA-TODO: CHECK better
                    // that expression is well formed
        ) {
          res += exprOp0Substr;
        } else {
          res += exprOp0;
        }

        goto GetExpr_end;
      } else { // i.e. if (exprOp0 == itVarPlusCt)
        // Case: *it is: x == phi(x + 1, 0);
        // Check getExpr(op0) == str0 + 1;

        LLVM_DEBUG(dbgs() << "    ... Entering getExpr() for op1\n");
        std::string exprOp1 = canonicalizeExpression(getExpr(op1));
        LLVM_DEBUG(dbgs() << "    exprOp1 = " << exprOp1 << "\n");
        // From NOW onwards we treat cases like
        //      for (c1=468; c1 >= 0; c1-=78)       assert(exprOp1 == "0");

        // assert(op0->getOpcode() == Instruction::ADD);
        /* assert that:
             - op1 is ct 0 and
             - op0 == iGetNameData + 1 (but this normally leads to
             a cyclic dependency)
           i.e., check that (str0 == iGetNameData) && (str1 == ct 0) */
        /* This next condition is Very important
         *   - e.g., for i phi node, for ...: because TODO
         */
        LLVM_DEBUG(
            dbgs()
            << "getExpr(): ...and str0 not empty, --> res = name of it\n");

        /* We don't modify iGetNameData - otherwise we get errors
        (assertion failures, etc) for modifying the LLVM variable names
        */
        strCopy.assign(iGetNameData);

        /* We might have a newly created temp LLVM var and keep the original
         (source file) variable name
        */
        strCopy = rStripStringAfterChar(strCopy, '.');
        res += strCopy;

        if (getExprVarSpecial) { // NOT tested, not sure if really required
          res += stringPrintf(const_cast<char *>("__%p"), (void *)it);
          // res += "<VAR*SPECIAL*END>";
        }

        goto GetExpr_end;
      }
    }
  } // end if ((it->getOpcode() == Instruction::PHI)

  // TODO: NOT sure if it's OK to only choose it->getOperand(0)
  // Normally this makes it a pointer to Value
  if (it->getNumOperands() == 0) {
    Type *itType = ((Value *)it)->getType();

    LLVM_DEBUG(dbgs() << "    (getExpr(): it->getType() = " << *itType
                      << " )\n");

    // See http://llvm.org/docs/doxygen/html/classllvm_1_1Type.html
    if (itType->isVectorTy()) {
      int64_t resVal = 0;

      // See http://llvm.org/docs/doxygen/html/classllvm_1_1ConstantVector.html
      /* Surprisingly NOT working:
          ConstantVector *ctVec = llvm::dyn_cast<ConstantVector>((Value *)it);
      */

      // See llvm.org/docs/doxygen/html/classllvm_1_1ConstantDataVector.html
      ConstantDataVector *ctVec = llvm::dyn_cast<ConstantDataVector>(it);

      LLVM_DEBUG(dbgs() << "getExpr(): ctVec =" << ctVec << "\n");

      if (ctVec != nullptr) {
        Constant *ctSplat = ctVec->getSplatValue();

        // See http://llvm.org/docs/doxygen/html/classllvm_1_1Constant.html
        const APInt ctAPInt = ctSplat->getUniqueInteger();
        // TODO: Use instead Constant::getAggregateElement() - see
        //  http://lists.llvm.org/pipermail/llvm-dev/2016-November/106954.html

        // See http://llvm.org/docs/doxygen/html/classllvm_1_1APInt.html
        resVal = ctAPInt.getSExtValue();
      }

      /* This was meant for the %induction vector var;
         but it's NOT good for %(broadcast).splatinsert - but we take
         care of this below ...TODO [SAY WHERE]
       */
      res += stringPrintf((char *)"(int)%ld", resVal);

      goto GetExpr_end;
    }

    // We print the constant or input variable:
    std::string Result;
    raw_string_ostream OS(Result);
    it->printAsOperand(OS, /* bool PrintType = */ false);
    OS.flush();
    LLVM_DEBUG(dbgs() << "    (getExpr(): it->printAsOperand() = " << Result
                      << ")\n");

    // We erase the leading % char if it exists - for name of var
    if (Result.c_str()[0] == '%')
      Result.erase(0, 1);

    // f16 (half) constants strangely are represented in LLVM in hexadecimal
    //   with prefix 0xH, as specified at http://llvm.org/docs/LangRef.html
    //   ("The IEEE 16-bit format (half precision) is represented by 0xH
    //      followed by 4 hexadecimal digits.")
    //   so now we remove the 'H' to make it proper hexadecimal value.
    if (startsWith(Result, "0xH"))
      Result.erase(2, 1);

    /*
    Result.clear();
    it->print(OS);
    OS.flush();
    LLVM_DEBUG(dbgs() << "    (getExpr(): it->print() = "
                      << Result << ")\n");
    */
    /*
    switch (it->getOpcode()) {
        case Instruction::Constant:
            LLVM_DEBUG(dbgs() << "    (getExpr(): it is Constant))\n");
            res = "ct";
            break;
    }
    */
    if (startsWith(Result, STR_UNDEF) == false) {
      /* Note:
         We can also have as parent
           %broadcast.splatinsert = insertelement <32 x i64> undef, i64 %mul.us,
                                    i32 0
         For this case, operand 0 is printed as: "<32 x i64> undef".
         But we avoid to reach this case by specially treating
          a %broadcast.splatinsert node.
      */
      res += Result;
    }
    goto GetExpr_end;
  } // End of if (it->getNumOperands() == 0)

  bool putParantheses;

  switch (it->getOpcode()) {
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::Trunc:
  case Instruction::ShuffleVector:
  case Instruction::InsertElement:
  case Instruction::PHI:
  case Instruction::ExtractElement:
    // res = "";
    putParantheses = false;
    break;

  case Instruction::GetElementPtr: {
    /*
    putParantheses = false;
    res = "(int *)&";
    */

    /* Important:
    From http://en.cppreference.com/w/c/language/operator_precedence:
        - operator [] (Array subscripting) has bigger priority
            than & (Address-of).
    So we need to put parantheses here
            in case [] follows.
    */
    putParantheses = true;

    // res = "(int *)&(";

    // By doing so we treat case like (&ls[index])[0] (see SSD benchmark)
    res = "((int *)&";

    getExprGEPCount++;
    getExprGEP = it;

    GetElementPtrInst *GEPInstr = llvm::dyn_cast<GetElementPtrInst>(it);
    assert(GEPInstr != nullptr);
    if (basePtrGetExprIt == nullptr)
      basePtrGetExprIt = GEPInstr->getPointerOperand();

    break;
  }
  default:
    putParantheses = true;
    res = "(";
  }
  /*
  if (putParantheses)
    res = "(";
  //
  if (it->getOpcode() == Instruction::GetElementPtr) { }
  */

  LLVM_DEBUG(dbgs() << "getExpr(): putParantheses = " << putParantheses
                    << "\n");

  if (it->getNumOperands() > 1) {
    LLVM_DEBUG(dbgs() << "getExpr(): it->getOperand(1) = " << *op1 << "; "
                      << "(str1 = " << str1 << ")[END]\n");

    // We prevent pretty-printing constant vectors
    // if (getExprForTripCount == false)
    /* TODO: maybe step.add is not operand 1, but 0 or 2, etc; check that
     op0 is constant */
    if (startsWith(iGetNameData, STR_VEC_IND) &&
        startsWith(str1, STR_STEP_ADD) &&
        startsWith(str0, STR_INDUCTION) == false) {
      /*
      This prevents further processing of:
          %vec.ind = phi <32 x i64> [ <i64 0, i64 1, ...>, %vector.ph ],
                                    [ %step.add, %vector.body ]
          BUT NOT of: %vec.ind = phi <32 x i32> [ %induction, %vector.ph ],
                                    [ %step.add, %vector.body ]
      */
      LLVM_DEBUG(
          dbgs()
          << "getExpr(): treating vec.ind = phi ct_vec, step.add case\n");

#ifdef AGGREGATED_DMA_TRANSFERS
      // Important note: we include this file from the back end also now
      //    (not only LoopVectorize.cpp)
      if (getExprForDMATransfer)
        res = "0";
      else
        res = "indexLLVM_LV";
#else
      if (getExprForDMATransfer)
        res = "0";
      else
        res = "indexLLVM_LV";
#endif
      goto GetExpr_end;
    }

    if (startsWith(iGetNameData, STR_VEC_IND) &&
        startsWith(str0, STR_INDUCTION) && startsWith(str1, STR_STEP_ADD)) {
      /*
      This prevents further processing of:
        %vec.ind = phi <32 x i64> [ <i64 0, i64 1, ...>, %vector.ph ],
                                  [ %step.add, %vector.body ]
        BUT NOT of: %vec.ind = phi <32 x i32> [ %induction, %vector.ph ],
                                              [ %step.add, %vector.body ]
      */
      LLVM_DEBUG(
          dbgs()
          << "getExpr(): treating vec.ind = phi induction, step.add case\n");
      res = getExpr(op0);
      res += " + indexLLVM_LV";
      goto GetExpr_end;
    }

    if (it->getOpcode() == Instruction::PHI) {
      // assert(0 && "We should not get here... since we already treated it");
      LLVM_DEBUG(
          dbgs() << "getExpr(): it is Phi. (normally should not be here)\n");
      LLVM_DEBUG(dbgs() << "  getExpr(): *it = " << *it << "\n");

      // Important-TODO : follow I guess the loopexit value
      /* This is for cases like the one encountered in 50_SpMV, where we
         cycle over temporary created vars:
              %1 = phi i16 [ %2, %for.cond.loopexit ],
                           [ %.pre, %for.body.preheader ]
              %2 = load i16, i16* %arrayidx5, align 2, !dbg !64, !tbaa !46
              %arrayidx5 = getelementptr inbounds i16, i16* %row_ptr,
                                                       i64 %idxprom4, !dbg !64
              %idxprom4 = sext i32 %add to i64, !dbg !64
              %add = add nsw i32 %i.026, 1, !dbg !63
              %i.026 = phi i32 [ %add, %for.cond.loopexit ],
                               [ 0, %for.body.preheader ]
      */
      res = getExpr(op0);
      LLVM_DEBUG(dbgs() << "getExpr(): it is Phi, res = " << res << "\n");

      /* Noname like in the case of 50_SpMV testcase:
          %1 = phi(%2, row_ptr[0])
      TODO But I guess I should check iGetName != str0 + 1...
      */
      // Note: constants like i64 0 don't have name --> str0 is empty
      if (str0.empty()) {
        LLVM_DEBUG(dbgs() << "getExpr(): it is Phi, str0 is empty.\n");

        // assert getNumOperands() > 1
        std::string res2 = getExpr(op1);
        LLVM_DEBUG(dbgs() << "getExpr(): res2 = " << res2 << "\n");
        // res += " phi ";

        /* Here we compute the solution of phi - a 1st simple and ~bad
         * attempt.
           MEGA-TODO: compute the
         closed-form solution from these recursive equations.
         */
        #define STR_TO_LOOK_FOR " + 1"
        res = canonicalizeExpression(res);
        LLVM_DEBUG(dbgs() << "getExpr(): After canonicalizeExpression() res = "
                          << res << "\n");
        std::size_t found = res.find(STR_TO_LOOK_FOR);
        if (found != std::string::npos) {
          LLVM_DEBUG(dbgs() << "getExpr(): calling res.erase(found, "
                               "strlen(STR_TO_LOOK_FOR))\n");
          res.erase(found, strlen(STR_TO_LOOK_FOR));
          res = canonicalizeExpression(res, true);
        }
        /*
        // BUGS: because of modifying the internal char * of a std::strng
        //        and I guess string::size() needs to be updated
        //        also(??)
        const char *resCStr = res.c_str();
        char *resCStrFound = (char *)strstr(resCStr, STR_TO_LOOK_FOR);
        if (resCStrFound != nullptr) {
          LLVM_DEBUG(dbgs() << "InstrumentVectorStore(): resCStrFound = "
                            << resCStrFound << "\n");
          // NOT correct - strings do overlap: strcpy(resCStrFound,
          //                                          resCStrFound + 4);
          memmove(resCStrFound, resCStrFound + strlen(STR_TO_LOOK_FOR),
                  strlen(resCStrFound + strlen(STR_TO_LOOK_FOR)) + 1);
        }
        */
      } else {
        /* Important-TODO: think if it is correct to be empty
          - try it out - note there is also another case treating
          phi nodes above.
        */
      }

      goto GetExpr_end;
    }

    /*
    // NOT necessary anymore - treat below this case by simply jumping to
    //      meaningful values
    if (startsWith(iGetNameData, STR_BROADCAST_SPLATINSERT) ||
        startsWith(iGetNameData, STR_SPLATINSERT)) {
      LLVM_DEBUG(dbgs()
                  << "getExpr(): treating (broadcast).splat(insert) case\n");

      // op0 should be vector undef
      res = getExpr(op1);
      goto GetExpr_end;
    }
    */
    if (startsWith(iGetNameData, STR_BROADCAST_SPLAT) ||
        /* // I guess it's not necessary to do this test:
           && (startsWith(iGetNameData, STR_BROADCAST_SPLATINSERT) == false) */
        startsWith(iGetNameData, STR_SPLAT)
        /* // I guess it's not necessary to do this test:
           && (startsWith(iGetNameData, STR_SPLATINSERT) == false) */
    ) {
      LLVM_DEBUG(dbgs() << "getExpr(): treating (broadcast).splat case\n");

      /* This is for the SSD test:
      %broadcast.splat33 = shufflevector <128 x i16> %broadcast.splatinsert32,
                           <128 x i16> undef, <128 x i32> zeroinitializer
      where it =
        %broadcast.splatinsert32 = insertelement <128 x i16> undef, i16 %0,
                                                             i32 0
      and op0 = <128 x i16> undef
      */
      if (llvm::dyn_cast<Instruction>(op0) == nullptr) {
        res = getExpr(op1);
        goto GetExpr_end;

        /*it->getOpcode() == Instruction::InsertElement)
          if (startsWith(iGetNameData, STR_BROADCAST_SPLAT) ||
        */
      } else {
        /// TODO: Maybe I should do some checks
        // op1 should be vector undef, op2 should be zeroinitializer
        // res = getExpr(op0);
        res = getExpr((((Instruction *)op0)->getOperand(1)));
        goto GetExpr_end;
      }
    }
  }

  // We now pretty print op0;

  if (str0.empty()
      /* ||
      startsWith(str0, STR_BROADCAST_SPLATINSERT) */
     ) {
    /* If the name of the variable is empty it means it is an automatically
     *   generated name (like %0, etc), NOT a name from the original (C,C++)
     *   program. Therefore we look also at the def of this var.
     */

    /*
    TODO
      - ~BAD: recursively test str0 until we reach a
          variable name that is input to the function??
    */
    /* TODO (THIS IS MAYBE BADLY DESIGNED - might require more or fewer steps):
     *  Coping with type conversions like i32 to i64 (ex:
     *    201_LoopVectorize/25_GOOD_map/NEW/7_v16i32/3better_opt.ll)
     *    in which case we have the following:
            for.body.preheader:                               ; preds = %entry
              %0 = add i32 %N, -1
              %1 = zext i32 %0 to i64
              %2 = add nuw nsw i64 %1, 1
              %min.iters.check = icmp ult i64 %2, 16
            [...]
            min.iters.checked:                   ; preds = %for.body.preheader
              %n.vec = and i64 %2, 8589934576
     */

    /*
    LLVM_DEBUG(dbgs() << "getExpr(): (it->getOperand(0) = "
                      << * (it->getOperand(0)) << ")\n");
    */
    LLVM_DEBUG(
        dbgs() << "getExpr(): str0 empty (or so) --> calling getExpr(op0)\n");
    LLVM_DEBUG(dbgs() << "    (getExpr(): current it = " << *it << ").\n");

    // strcpy(res, tmp);
    res += getExpr(op0);
  } else { // str0 is NOT empty
    /*
    if (getExprForTripCount == false) {
      LLVM_DEBUG(dbgs() << "getExpr(): returning str0 = "
                   << str0 << "\n");
      //res.assign(str0);
      // Gives <<warning: cast from type ‘const char*’ to type ‘char*’ casts
      // away qualifiers>>
      // * (char *)strchr(str0, '.') = 0;

      // Important-TODO: this
      //      transformation I guess is NOT 100% safe, because a named var
      //      can be a C var or an auxiliary LLVM var created in the LLVM pass
      //      - think how to make it safe

      if (startsWith(str0, STR_VEC_IND) == false) {
        // We don't modify str0 - otherwise we get errors
        //(assertion failures, etc) for modifying the LLVM variable names
        strCopy.assign(str0);

        // We might have a newly created temp LLVM var and keep the original
        // (source file) variable name
        strCopy = rStripStringAfterChar(strCopy, '.');
        res += strCopy;

        // Maybe put here operation pretty-print TODO
      }
      else {
        // vec.ind is the widened induction variable
        // res += str0;
      }
    }
    else
    */
    { // getExprForTripCount == true and str0 not empty
      /*
      // This SOMETIMES introduces infinite cycles, which can be avoided
      // if we keep track of the instructions already visited
      Example of cycle:
        - these 2 simple instructions:
          %indvars.iv29 = phi i64 [ 0, %for.body.preheader ],
                                  [ %indvars.iv.next30, %for.cond.loopexit ].
          %indvars.iv.next30 = add nuw nsw i64 %indvars.iv29, 1, !dbg !9
      */

      if ((it->getOpcode() == Instruction::GetElementPtr) &&
          (it->getNumOperands() >= 3)) {
        res += ((Instruction *)op0)->getName().data();
      } else {
        LLVM_DEBUG(
            dbgs() << "getExpr(): str0 not empty --> calling getExpr(op0)\n");
        // This introduces useless parantheses: res += "(";
        res += getExpr(op0);
        // This introduces useless parantheses: res += ")";
      }
    }
  }

  // We now pretty print operation associated to *it;

  /* We generate C code for the operation associated to the it
     LLVM instruction.
   See http://llvm.org/docs/doxygen/html/Instruction_8cpp_source.html
      for all/various possible opcodes - see method
         00194 const char *Instruction::getOpcodeName(unsigned OpCode). */
  // Note: vec.ind is a PHI node
  // if (startsWith(str0, STR_VEC_IND) == false)
  // if (!(getExprForTripCount == false && strcmp(str0, "vec.ind") == 0))
  switch (it->getOpcode()) {
  case Instruction::Call: {
    // Important-TODO: this works well for the case 31c_dotprod_RaduH,
    //                    BUT not sure if it's general
    res = "((int *)&(";
    res += iGetNameData;
    res += "))";

    std::string strFuncName;
    strFuncName = dyn_cast<CallInst>(it)->getCalledFunction()->getName().str();
    assert(strFuncName == "malloc" || strFuncName == "calloc");

    // Inspired from llvm.org/docs/ProgrammersManual.html
    //          #iterating-over-def-use-use-def-chains
    for (Value::user_iterator i = it->user_begin(), e = it->user_end(); i != e;
         ++i) {
      if (Instruction *inst = dyn_cast<Instruction>(*i)) {
        LLVM_DEBUG(dbgs() << "getExpr(): it is used in instruction: " << *inst
                          << "\n");
        if (BitCastInst *bci = dyn_cast<BitCastInst>(*i)) {
          if (strlen(bci->getName().data()) != 0) {
            LLVM_DEBUG(
                dbgs()
                << "getExpr(): it is used in BitCast instruction --> we use "
                   "its name instead\n");
            res = "((int *)&(";
            res += bci->getName().data();
            res += "))";
          } else {
            if (ranGetAllMetadata == false) {
              LLVM_DEBUG(dbgs() << "getExpr(): Before, varNameMap.size() = "
                                << varNameMap.size() << "\n");
              getAllMetadata(bci->getParent()->getParent());
              LLVM_DEBUG(dbgs() << "getExpr(): varNameMap.size() = "
                                << varNameMap.size() << "\n");
            }

            std::string valueName = getLLVMValueName(bci);

            // Normally the value name is a number when getName() is empty
            LLVM_DEBUG(dbgs() << "getExpr(): bci has empty name\n");
            LLVM_DEBUG(dbgs() << "getExpr(): bci = " << *bci << "\n");
            LLVM_DEBUG(dbgs() << "           bci = " << bci << "\n");
            LLVM_DEBUG(dbgs() << "           bci->getValueName() = "
                              << bci->getValueName() << "\n");
            LLVM_DEBUG(dbgs() << "           bci->getName() = "
                              << bci->getName() << "\n");
            LLVM_DEBUG(dbgs() << "getExpr(): it = " << *it << "\n");
            //
            LLVM_DEBUG(dbgs() << "getExpr(): varNameMap[bci] = "
                              << varNameMap[valueName] << "\n");

            // res = varNameMap[valTypeAndName];
            res = varNameMap[valueName];

            goto GetExpr_end;

            /*
            for (Value::user_iterator i2 = bci->user_begin(),
                                      e2 = bci->user_end();
                                      i2 != e2; ++i2) {
              if (Instruction *inst2 = dyn_cast<Instruction>(*i2)) {
                LLVM_DEBUG(dbgs() << "getExpr(): bci is used in instruction: "
                             << *inst2 << "\n");
                if (StoreInst *si = dyn_cast<StoreInst>(*i2)) {
                  LLVM_DEBUG(dbgs()
                    << "getExpr(): bci is used in StoreInst instruction "
                       "--> we use its name instead\n");
                  res = "((int *)&(";
                  res += si->getName().data();
                  res += "))";
                  goto GetExpr_end;
                }
              }
            }
            */
          }
        }
      } else {
        LLVM_DEBUG(dbgs() << "getExpr(): it is used in val: " << *i << "\n");
      }
    }

    goto GetExpr_end;
  }
  case Instruction::Add:
  case Instruction::FAdd:
    res += " + ";
    break;
  case Instruction::Sub:
    res += " - ";
    break;
  // case Instruction::FSub:
  case Instruction::Mul: {
    res += " * ";
    break;
  }
  // case Instruction::FMul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
    res += " / ";
    break;
  case Instruction::URem:
  case Instruction::SRem:
    // case Instruction::FRem:
    res += " % ";
    break;
  case Instruction::Shl:
    res += " << ";
    break;
  case Instruction::LShr:
    res += " >> ";
    break;
  // Important-TODO: think better
  case Instruction::AShr:
    /* From https://en.wikipedia.org/wiki/Arithmetic_shift#cite_ref-1 :
    "The >> operator in C and C++ is
       not necessarily an arithmetic shift. Usually it is only an
    arithmetic shift if used with a signed integer type on its
    left-hand side.
      If it is used on an unsigned integer type instead, it will be a
    logical shift."
    */
    res += " >> ";
    break;
  case Instruction::And:
    res += " & ";
    break;
  case Instruction::Or:
    res += " | ";
    break;
  case Instruction::Xor:
    res += " ^ ";
    break;
  case Instruction::PHI:
    res += " phi ";
    break;
  case Instruction::Load:
    // res += " load ";
    res += "[0]";
    break;
  case Instruction::Store:
    res += " store ";
    break;
  case Instruction::GetElementPtr:
    // res += " getelementptr ";
    /*
    if (it->getNumOperands() < 3) {
        res += " + ";
    }
    */
    break;
  case Instruction::ZExt:
  case Instruction::SExt:
    // res += " ext "; // Note: this is unary operator
    break;
  // case Instruction::FPTrunc:
  case Instruction::Trunc: {
    // res += " trunc ";
    break;
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    // Check type of cmp
    CmpInst *Cmp = dyn_cast<CmpInst>(it);
    int pred = Cmp->getPredicate();
    res += getPredicateString(pred);

    break;
  }
  case Instruction::Select: {
    // TODO: add : and 3rd operand
    res += " ? ";
    break;
  }
  case Instruction::ShuffleVector: {
    // res += " shufflevector ";
    break;
  }
  case Instruction::InsertElement: {
    // res += " insertelement ";
    break;
  }
  case Instruction::ExtractElement: {
    // res += " extractelement ";

    LLVM_DEBUG(dbgs() << "getExpr(): case Instruction::ExtractElement.\n");
    LLVM_DEBUG(dbgs() << "getExpr(): res = " << res << "\n");

    std::string op1Expr = getExpr(op1);
    // ((Instruction *)op1)->getName().data();

    // std::string op0Expr = getExpr(op0);

    if (op1Expr == "0") {
      LLVM_DEBUG(
          dbgs()
          << "getExpr(): Neutralizing ExtractElement, since index is 0\n");
      if (putParantheses)
        res += ")";

      goto GetExpr_end;
    }

    // TODO: check that op0 is vec.ind or sext vec.ind
    res = "((int *)&" + res;
    // res = "((int *)&" + op0Expr;
    res += "))"; // One ')' for the '(' added at beginning getExpr,
                 //   1 to close the '(' before '&'
    res += "[";
    res += op1Expr;
    res += "]";

    // basePtr = nullptr;

    goto GetExpr_end;
    // break;
  }
  // See e.g. http://llvm.org/docs/doxygen/html/Instructions_8h_source.html
  case Instruction::PtrToInt:
  case Instruction::IntToPtr: {
    /* This is normally encountered when using the LLVM-SRA library and
     I give SCEVRangeBuilder->getUpperBound(AccessFunction) */
    // We don't do a thing
    break;
  }
  case Instruction::Alloca: {
    // res += "(int *)&(";
    // TODO: this works well for the case 31c_dotprod_RaduH
    res = "((int *)&(";
    res += iGetNameData;
    res += "))";
    goto GetExpr_end;
    // break;
  }
  default:
    assert(0 && "NOT implemented");
  } // end switch

  /*
  if (it->getOpcode() == Instruction::PHI) {
    // TODO: check that op0 is associated to predecessor BB
    // different than itself - e.g., preheader, vector.ph, etc

    // This results in incorrect paranthesis - missing a few ')'
    goto GetExpr_end;
  }
  */

  // Pretty print op1:

  /*
  if ((it->getNumOperands() > 1) &&
      (it->getOpcode() != Instruction::PHI))
  */
  if (it->getNumOperands() > 1) {
    // strcat(res, " ");
    res += " ";

    bool specialCase = false;
    bool str1NotEmpty = (str1.empty() == false);

    if (str1NotEmpty) {
      LLVM_DEBUG(dbgs() << "getExpr(): str1 NOT empty: str1 = " << str1
                        << "\n");

      /* Important Note: some operands have names and are also
       instructions */

      /*
      The following can also introduce cycles:
        - an example
          getExpr(): str0 empty (or so) --> calling getExpr(op0)
              (getExpr(): current it =   %vec.ind = phi <32 x i64> [ <i64 0,
                i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9,
                i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16, i64 17,
                i64 18, i64 19, i64 20, i64 21, i64 22, i64 23, i64 24, i64 25,
           i64 26, i64 27, i64 28, i64 29, i64 30, i64 31>, %vector.ph ],
              [ %step.add, %vector.body ]).
          getExpr(): getExprForTripCount = 1
          getExpr(): it = <32 x i64> <i64 0, i64 1, i64 2, i64 3,
                      i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10,
                      i64 11, i64 12, i64 13, i64 14, i64 15, i64 16,
                      i64 17, i64 18, i64 19, i64 20, i64 21, i64 22,
                      i64 23, i64 24, i64 25, i64 26, i64 27, i64 28,
                      i64 29, i64 30, i64 31>
              (getExpr(): it->getOpcodeName() = <Invalid operator> )
              (getExpr(): it->getName() = )
              (getExpr(): it->printAsOperand() == <i64 0, i64 1, i64 2, i64 3,
                i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11,
                i64 12, i64 13, i64 14, i64 15, i64 16, i64 17, i64 18, i64 19,
                i64 20, i64 21, i64 22, i64 23, i64 24, i64 25, i64 26, i64 27,
                i64 28, i64 29, i64 30, i64 31>)
          getExpr(): calling getExpr(op1).
              getExpr(): it =   %vec.ind = phi <32 x i64> [ <i64 0, i64 1,
                i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10,
                i64 11, i64 12, i64 13, i64 14, i64 15, i64 16, i64 17, i64 18,
                i64 19, i64 20, i64 21, i64 22, i64 23, i64 24, i64 25, i64 26,
          i64 27, i64 28, i64 29, i64 30, i64 31>, %vector.ph ],
              [ %step.add, %vector.body ].
          getExpr(): getExprForTripCount = 1
          getExpr(): it =   %step.add = add <32 x i64> %vec.ind,
                          <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32,
                          i64 32, i64 32, i64 32, i64 32, i64 32, i64 32,
                          i64 32, i64 32, i64 32, i64 32, i64 32, i64 32,
                          i64 32, i64 32, i64 32, i64 32, i64 32, i64 32,
                          i64 32, i64 32, i64 32, i64 32, i64 32, i64 32,
                          i64 32, i64 32>, !dbg !38

         - another example:
          getExpr(): it =   %row.020.us = phi i64 [ %inc16.us,
                                   %for.cond1.for.cond.cleanup3_crit_edge.us ],
                                   [ 0, %for.cond1.preheader.us.preheader ]
              (getExpr(): it->getOpcodeName() = phi)
              (getExpr(): it->getName() = row.020.us)
          getExpr(): it->getOperand(1) = i64 0; str1 = [END]
          getExpr(): getExprForTripCount = 1
          getExpr(): it =   %inc16.us = add nuw nsw i64 %row.020.us, 1, !dbg !58
              (getExpr(): it->getOpcodeName() = add)
              (getExpr(): it->getName() = inc16.us)
          getExpr(): it->getOperand(1) = i64 1; str1 = [END]
          getExpr(): getExprForTripCount = 1
          getExpr(): it =   %row.020.us = phi i64
                       [ %inc16.us, %for.cond1.for.cond.cleanup3_crit_edge.us ],
                       [ 0, %for.cond1.preheader.us.preheader ]
      */
      // if (strcmp(str1, "broadcast.splat") == 0)
      if (((Instruction *)op1)->getNumOperands() != 0 &&
          // This prevents pretty-printing constant vectors, etc
          !(startsWith(iGetNameData, STR_VEC_IND) &&
            startsWith(str1, STR_STEP_ADD))) {
        // res += getExpr(op1);

        // We defer pretty printing below - see immediately below
        /*
        LLVM_DEBUG(dbgs() << "getExpr(): calling getExpr(op1).\n");
        LLVM_DEBUG(dbgs() << "    getExpr(): it = " << *it << ".\n");
        res += getExpr(op1);
        */
      } else {
        res += str1;
        specialCase = true;
      }
    } // End str1 NOT empty

    if (specialCase == false) {
      LLVM_DEBUG(dbgs() << "getExpr(): specialCase = false, "
                        << "str1NotEmpty = " << str1NotEmpty << ".\n");
      LLVM_DEBUG(dbgs() << "    getExpr(): it = " << *it << ".\n");

      if (it->getOpcode() == Instruction::GetElementPtr) {
        int numOpnds = it->getNumOperands();
        GetElementPtrInst *GEPPtr = llvm::dyn_cast<GetElementPtrInst>(it);
        int startIndex = getIndexFirstOpndFromGEPInst(GEPPtr);

        for (int i = startIndex; i < numOpnds; i++) {
          res += "[";

          Value *op_i = it->getOperand(i);
          res += getExpr(op_i);

          res += "]";
        }
      } else {
        // strcat(res, getExpr(op1));
        res += getExpr(op1);
      }
    }
  } // END if (it->getNumOperands() > 1)

  // Important-TODO : treat also Phi, which can have arbitrary num of arguments:
  // if (it->getOpcode() == Instruction::Phi)
  if (it->getOpcode() == Instruction::Select) {
    // ConstantExpr: || isSelectConstantExpr)
    res += " : ";

    Value *op2;
    /*
    // ConstantExpr:
    LLVM_DEBUG(dbgs()
              << "getExpr(): isSelectConstantExpr = "
              << isSelectConstantExpr);

    if (isSelectConstantExpr) {
      op2 = CE->getOperand(2);
    }
    else {
    */
    op2 = it->getOperand(2);
    //}
    LLVM_DEBUG(dbgs() << "getExpr(): *op2 = " << *op2);
    res += getExpr(op2);
  }

  if (putParantheses)
    res += ")";

GetExpr_end:
  /*
  // Don't really understand why it fails at compile-time at make_pair
      // std::unordered_map<Value *, std::string> cacheExpr;
      typedef Value *ValuePtr;
      //cacheExpr.insert(std::make_pair<Value *, std::string>(it, res));
      cacheExpr.insert(std::make_pair<InstructionPtr, std::string>(it, res));
  But this does NOT fail:
      // Inspired from example
      //               http://www.cplusplus.com/reference/utility/make_pair/
      std::pair<Value *, std::string> tmp;
      tmp = std::make_pair(it, res);
      cacheExpr.insert(tmp);
  */
  /*
  if ((res.size() == 2) && (res.c_str()[0] == '(') &&
          (res.c_str()[1] == ')'))
  */
  if (res == "()") {
    // This is redundant so we drop it.
    res.clear();
  }

  LLVM_DEBUG(dbgs() << "getExpr(): Inserting in cacheExpr aVal = " << aVal;
             if (aVal != nullptr) dbgs()
             << " (*aVal = " << *aVal << ") and res = " << res;
             dbgs() << "\n");
  cacheExpr[aVal] = res;

  return res;
} // end getExpr()

} // end namespace

#endif // RECOVER_FROM_LLVM_IR
