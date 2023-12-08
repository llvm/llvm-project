// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class ValueType<int size, int value> {
  int Size = size;
  int Value = value;
}

def f32  : ValueType<32, 1>;   //  2 x i64 vector value

class Intrinsic<string name> {
  string Name = name;
}

class Inst<bits<8> opcode, dag oopnds, dag iopnds, string asmstr, 
           list<dag> pattern> {
  bits<8> Opcode = opcode;
  dag OutOperands = oopnds;
  dag InOperands = iopnds;
  string AssemblyString = asmstr;
  list<dag> Pattern = pattern;
}

def ops;
def outs;
def ins;

def set;

// Define registers
class Register<string n> {
  string Name = n;
}

class RegisterClass<list<ValueType> regTypes, list<Register> regList> {
  list<ValueType> RegTypes = regTypes;
  list<Register> MemberList = regList;
}

def XMM0: Register<"xmm0">;
def XMM1: Register<"xmm1">;
def XMM2: Register<"xmm2">;
def XMM3: Register<"xmm3">;
def XMM4: Register<"xmm4">;
def XMM5: Register<"xmm5">;
def XMM6: Register<"xmm6">;
def XMM7: Register<"xmm7">;
def XMM8:  Register<"xmm8">;
def XMM9:  Register<"xmm9">;
def XMM10: Register<"xmm10">;
def XMM11: Register<"xmm11">;
def XMM12: Register<"xmm12">;
def XMM13: Register<"xmm13">;
def XMM14: Register<"xmm14">;
def XMM15: Register<"xmm15">;

def FR32 : RegisterClass<[f32],
                         [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                          XMM8, XMM9, XMM10, XMM11,
                          XMM12, XMM13, XMM14, XMM15]>;

class SDNode {}
def not : SDNode;

multiclass scalar<bits<8> opcode, string asmstr = "", list<list<dag>> patterns = []> {
  def SSrr : Inst<opcode, (outs FR32:$dst), (ins FR32:$src),
                  !strconcat(asmstr, "\t$dst, $src"),
                  !if(!empty(patterns),[]<dag>,patterns[0])>;
  def SSrm : Inst<opcode, (outs FR32:$dst), (ins FR32:$src),
                  !strconcat(asmstr, "\t$dst, $src"),
                  !if(!empty(patterns),[]<dag>,!if(!empty(!tail(patterns)),patterns[0],patterns[1]))>;
}

multiclass vscalar<bits<8> opcode, string asmstr = "", list<list<dag>> patterns = []> {
  def V#NAME#SSrr : Inst<opcode, (outs FR32:$dst), (ins FR32:$src),
                  !strconcat(asmstr, "\t$dst, $src"),
                  !if(!empty(patterns),[]<dag>,patterns[0])>;
  def V#NAME#SSrm : Inst<opcode, (outs FR32:$dst), (ins FR32:$src),
                  !strconcat(asmstr, "\t$dst, $src"),
                  !if(!empty(patterns),[]<dag>,!if(!empty(!tail(patterns)),patterns[0],patterns[1]))>;
}

multiclass myscalar<bits<8> opcode, string asmstr = "", list<list<dag>> patterns = []> :
  scalar<opcode, asmstr, patterns>,
  vscalar<opcode, asmstr, patterns>;

defm NOT : myscalar<0x10, "not", [[], [(set FR32:$dst, (f32 (not FR32:$src)))]]>;

// CHECK: Pattern = [(set FR32:$dst, (f32 (not FR32:$src)))];
// CHECK: Pattern = [];
// CHECK: Pattern = [(set FR32:$dst, (f32 (not FR32:$src)))];
// CHECK: Pattern = [];
