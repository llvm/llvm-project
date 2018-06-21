// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class ValueType<int size, int value> {
  int Size = size;
  int Value = value;
}

def v2i64  : ValueType<128, 22>;   //  2 x i64 vector value
def v2f64  : ValueType<128, 28>;   //  2 x f64 vector value

class Intrinsic<string name> {
  string Name = name;
}

class Pattern<dag patternToMatch, list<dag> resultInstrs> {
  dag             PatternToMatch  = patternToMatch;
  list<dag>       ResultInstrs    = resultInstrs;
}

// Pat - A simple (but common) form of a pattern, which produces a simple result
// not needing a full list.
class Pat<dag pattern, dag result> : Pattern<pattern, [result]>;

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

def VR128 : RegisterClass<[v2i64, v2f64],
                          [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                           XMM8, XMM9, XMM10, XMM11,
                           XMM12, XMM13, XMM14, XMM15]>;

// Dummy for subst
def REGCLASS : RegisterClass<[], []>;
def MNEMONIC;

class decls {
  // Dummy for foreach
  dag pattern;
  int operand;
}

def Decls : decls;

// Define intrinsics
def int_x86_sse2_add_ps : Intrinsic<"addps">;
def int_x86_sse2_add_pd : Intrinsic<"addpd">;
def INTRINSIC : Intrinsic<"Dummy">;
def bitconvert;

class MakePat<list<dag> patterns> : Pat<patterns[0], patterns[1]>;

class Base<bits<8> opcode, dag opnds, dag iopnds, string asmstr, Intrinsic intr, 
           list<list<dag>> patterns>
      : Inst<opcode, opnds, iopnds, asmstr, 
             !foreach(Decls.pattern, patterns[0], 
		      !foreach(Decls.operand, Decls.pattern, 
			       !subst(INTRINSIC, intr, 
		               !subst(REGCLASS, VR128, 
                               !subst(MNEMONIC, set, Decls.operand)))))>,
        MakePat<!foreach(Decls.pattern, patterns[1], 
		         !foreach(Decls.operand, Decls.pattern, 
			          !subst(INTRINSIC, intr, 
				  !subst(REGCLASS, VR128, 
                                  !subst(MNEMONIC, set, Decls.operand)))))>;

multiclass arith<bits<8> opcode, string asmstr, string intr, list<list<dag>> patterns> {
  def PS : Base<opcode, (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                 !strconcat(asmstr, "\t$dst, $src1, $src2"), !cast<Intrinsic>(!subst("SUFFIX", "_ps", intr)), patterns>;

  def PD : Base<opcode, (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                 !strconcat(asmstr, "\t$dst, $src1, $src2"), !cast<Intrinsic>(!subst("SUFFIX", "_pd", intr)), patterns>;
}

defm ADD : arith<0x58, "add", "int_x86_sse2_addSUFFIX",
                  // rr Patterns
                 [[(set REGCLASS:$dst, (INTRINSIC REGCLASS:$src1, REGCLASS:$src2))],
                   [(set REGCLASS:$dst, (bitconvert (INTRINSIC REGCLASS:$src1, REGCLASS:$src2))),
                    (MNEMONIC REGCLASS:$dst, REGCLASS:$src)]]>;

// CHECK: [(set VR128:$dst, (int_x86_sse2_add_pd VR128:$src1, VR128:$src2))]
// CHECK: [(set VR128:$dst, (int_x86_sse2_add_ps VR128:$src1, VR128:$src2))]
