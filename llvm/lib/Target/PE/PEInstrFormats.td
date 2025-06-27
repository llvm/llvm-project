class PEInst<dag outs, dag ins, string asmstr,list<dag> pattern>
: Instruction
{
    field bits<32> SoftFail = 0;//软件错误，传0，用于反汇编程序

    let Size = 4;
    let Namespace = "PE";
    dag OutOperandList = outs;
    dag InOperandList = ins;
    let AsmString = asmstr;
    let Pattern = pattern;
}
//伪指令
class PseudoInst<dag outs, dag ins,list<dag> pattern,string asmstr = "">
    : PEInst<outs, ins, asmstr, pattern> {
  let isPseudo = 1;//表示为伪指令
  let isCodeGenOnly = 1;//存在于代码生成
  let Size = 4;
}
class I<dag outs, dag ins,
              string asmstr, list<dag> pattern>
    : PEInst< outs, ins, asmstr, pattern> {
}
class U<dag outs, dag ins, 
              string asmstr, list<dag> pattern>
    : PEInst<outs, ins, asmstr, pattern> {
  // 这里可以加U型特有的属性
}
class B<dag outs, dag ins, string asm, list<dag> pattern> : 
			PEInst<outs, ins, asm, pattern> {

      }
class J<dag outs, dag ins, string asm, list<dag> pattern> : 
			PEInst<outs, ins, asm, pattern> {

      }
class S<dag outs, dag ins, string asm, list<dag> pattern> : 
			PEInst<outs, ins, asm, pattern> {

      }