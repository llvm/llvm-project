//寄存器信息
//bit<5>代表5bit给寄存器标号,string n：寄存器名字，alt：寄存器别名
class PEReg<bits<5> Enc, string n, list<string> alt = []> :
Register<n> {
    let HWEncoding{4-0} = Enc;//后5位为寄存器编码
    let Namespace = "PE";
    let AltNames = alt;
}
def ABIRegAltName : RegAltNameIndex;

//描述寄存器信息
let RegAltNameIndices = [ABIRegAltName] in {
  let isConstant = true in{
    def X0  : PEReg<0, "x0", ["zero"]>;
  }
 
  def X1  : PEReg<1, "x1", ["ra"]>;
  def X2  : PEReg<2, "x2", ["sp"]>;
 
  def X10  : PEReg<10, "x10", ["a0"]>;
  def X11  : PEReg<11, "x11", ["a1"]>;
}

def GPR : RegisterClass<"PE",[i32],32,(add X0,X1,X10,X11)>;