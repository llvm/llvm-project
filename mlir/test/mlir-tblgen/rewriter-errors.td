// RUN: not mlir-tblgen -gen-rewriters -I %S/../../include -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not mlir-tblgen -gen-rewriters -I %S/../../include -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not mlir-tblgen -gen-rewriters -I %S/../../include -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// RUN: not mlir-tblgen -gen-rewriters -I %S/../../include -DERROR4 %s 2>&1 | FileCheck --check-prefix=ERROR4 %s
// RUN: not mlir-tblgen -gen-rewriters -I %S/../../include -DERROR5 %s 2>&1 | FileCheck --check-prefix=ERROR5 %s
// RUN: not mlir-tblgen -gen-rewriters -I %S/../../include -DERROR6 %s 2>&1 | FileCheck --check-prefix=ERROR6 %s
// RUN: not mlir-tblgen -gen-rewriters -I %S/../../include -DERROR7 %s 2>&1 | FileCheck --check-prefix=ERROR7 %s

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

// Check using the dialect name as the namespace
def A_Dialect : Dialect {
  let name = "a";
}

class A_Op<string mnemonic, list<Trait> traits = []> :
    Op<A_Dialect, mnemonic, traits>;

def OpA : A_Op<"op_a">, Arguments<(ins AnyInteger, AnyInteger)>, Results<(outs AnyInteger)>;
def OpB : A_Op<"op_b">, Arguments<(ins AnyInteger, AnyAttr:$value)>, Results<(outs AnyInteger)>;

#ifdef ERROR1
def NativeMatcher : NativeCodeCall<"success(nativeCall($0, $1))">;
// ERROR1: [[@LINE+1]]:1: error: NativeCodeCall must have $_self as argument for passing the defining Operation
def : Pat<(OpA (NativeMatcher $val), AnyI32Attr:$arg),
          (OpB $val, $arg)>;
#endif

#ifdef ERROR2
def NativeMatcher : NativeCodeCall<"success(nativeCall($_self, &$0))">;
// ERROR2: [[@LINE+1]]:1: error: binding symbol 'error' to NativecodeCall in MatchPattern is not supported
def : Pat<(OpA (NativeMatcher:$error $val), AnyI32Attr:$arg),
          (OpB $val, $arg)>;
#endif

#ifdef ERROR3
def NativeMatcher : NativeCodeCall<"success(nativeCall($_self, $0, $1))">;
// ERROR3: [[@LINE+1]]:1: error: Matching nested tree in NativeCodecall not support for
def : Pat<(OpA (NativeMatcher (OpB $val, $unused)), AnyI32Attr:$arg),
          (OpB $val, $arg)>;
#endif

#ifdef ERROR4
// Check trying to pass op as DAG node inside ReturnTypeFunc fails.
// ERROR4: [[@LINE+1]]:1: error: nested DAG in `returnType` must be a native code
def : Pat<(OpB $val, AnyI32Attr:$attr), (OpA (OpA $val, $val, (returnType (OpA $val, $val))), $val)>;
#endif

#ifdef ERROR5
// Check that trying to specify explicit types at the root node fails.
// ERROR5: [[@LINE+1]]:1: error: Cannot specify explicit return types in an op
def : Pat<(OpB $val, AnyI32Attr:$attr), (OpA $val, $val, (returnType "someType()"))>;
#endif

#ifdef ERROR6
// Check that type constraint has one argument
// ERROR6: [[@LINE+1]]:1: error: type constraint requires exactly one argument
def : Pat<(OpB:$result $val, $attr), (OpA $val, $val), [(AnyInteger:$result)]>;
#endif

#ifdef ERROR7
// Check that type constraint has one argument
// ERROR7: [[@LINE+1]]:1: error: type constraint requires exactly one argument
def : Pat<(OpB:$opB $val, $attr), (OpA $val, $val), [(AnyInteger $opB, $val)]>;
#endif
