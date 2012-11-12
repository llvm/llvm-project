// RUN: llvm-tblgen %s | grep "add_ps" | count 3
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

// Define intrinsics
def int_x86_sse2_add_ps : Intrinsic<"addps">;
def int_x86_sse2_add_pd : Intrinsic<"addpd">;

multiclass arith<bits<8> opcode, string asmstr, string Intr> {
  def PS : Inst<opcode, (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                 !strconcat(asmstr, "\t$dst, $src1, $src2"),
                 [(set VR128:$dst, (!cast<Intrinsic>(!strconcat(Intr, "_ps")) VR128:$src1, VR128:$src2))]>;

  def PD : Inst<opcode, (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                 !strconcat(asmstr, "\t$dst, $src1, $src2"),
                 [(set VR128:$dst, (!cast<Intrinsic>(!strconcat(Intr, "_pd")) VR128:$src1, VR128:$src2))]>;
}

defm ADD : arith<0x58, "add", "int_x86_sse2_add">;

class IntInst<bits<8> opcode, string asmstr, Intrinsic Intr> :
  Inst<opcode,(outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
       !strconcat(asmstr, "\t$dst, $src1, $src2"),
       [(set VR128:$dst, (Intr VR128:$src1, VR128:$src2))]>;


multiclass arith_int<bits<8> opcode, string asmstr, string Intr> {
  def PS_Int : IntInst<opcode, asmstr, !cast<Intrinsic>(!strconcat(Intr, "_ps"))>;

  def PD_Int : IntInst<opcode, asmstr, !cast<Intrinsic>(!strconcat(Intr, "_pd"))>;
}

defm ADD : arith_int<0x58, "add", "int_x86_sse2_add">;
