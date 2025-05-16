
class PEReg<string n> :
Register<n> {
    let Namespace = "PE";
}


def ZERO : PEReg<"zero">,DwarfRegNum<[0]>;
def RA : PEReg<"ra">,DwarfRegNum<[1]>;
def SP : PEReg<"sp">,DwarfRegNum<[2]>;
def T0 : PEReg<"t0">,DwarfRegNum<[5]>;
def T1 : PEReg<"t1">,DwarfRegNum<[6]>;
def T2 : PEReg<"t2">,DwarfRegNum<[7]>;
def S0 : PEReg<"s0">,DwarfRegNum<[8]>;
def S1 : PEReg<"s1">,DwarfRegNum<[9]>;
def A0 : PEReg<"a0">,DwarfRegNum<[10]>;
def A1 : PEReg<"a1">,DwarfRegNum<[11]>;


def GPR : RegisterClass<"PE",[i32],32,(add 
    ZERO,RA,SP,
    T0,T1,T2,
    S0,S1,
    A0,A1
)>;