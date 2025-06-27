
// class PEReg<string n> :
// Register<n> {
//     let Namespace = "PE";
// }


// def ZERO : PEReg<"zero">,DwarfRegNum<[0]>;
// def RA : PEReg<"ra">,DwarfRegNum<[1]>;
// def SP : PEReg<"sp">,DwarfRegNum<[2]>;
// def T0 : PEReg<"t0">,DwarfRegNum<[5]>;
// def T1 : PEReg<"t1">,DwarfRegNum<[6]>;
// def T2 : PEReg<"t2">,DwarfRegNum<[7]>;
// def S0 : PEReg<"s0">,DwarfRegNum<[8]>;
// def S1 : PEReg<"s1">,DwarfRegNum<[9]>;
// def A0 : PEReg<"a0">,DwarfRegNum<[10]>;
// def A1 : PEReg<"a1">,DwarfRegNum<[11]>;


// def GPR : RegisterClass<"PE",[i32],32,(add 
//     ZERO,RA,SP,
//     A0,A1,
//     T0,T1,T2,
//     S0,S1
// )>;

class PEReg<string n> : Register<n> {
    let Namespace = "PE";
}

// 标量寄存器 RS0 ~ RS27
def RS0  : PEReg<"RS0">,  DwarfRegNum<[0]>;   
def RS1  : PEReg<"RS1">,  DwarfRegNum<[1]>;
def RS2  : PEReg<"RS2">,  DwarfRegNum<[2]>;
def RS3  : PEReg<"RS3">,  DwarfRegNum<[3]>;
def RS4  : PEReg<"RS4">,  DwarfRegNum<[4]>;
def RS5  : PEReg<"RS5">,  DwarfRegNum<[5]>;
def RS6  : PEReg<"RS6">,  DwarfRegNum<[6]>;
def RS7  : PEReg<"RS7">,  DwarfRegNum<[7]>;
def RS8  : PEReg<"RS8">,  DwarfRegNum<[8]>;
def RS9  : PEReg<"RS9">,  DwarfRegNum<[9]>;
def RS10 : PEReg<"RS10">, DwarfRegNum<[10]>;
def RS11 : PEReg<"RS11">, DwarfRegNum<[11]>;
def RS12 : PEReg<"RS12">, DwarfRegNum<[12]>;
def RS13 : PEReg<"RS13">, DwarfRegNum<[13]>;
def RS14 : PEReg<"RS14">, DwarfRegNum<[14]>;
def RS15 : PEReg<"RS15">, DwarfRegNum<[15]>;
def RS16 : PEReg<"RS16">, DwarfRegNum<[16]>;
def RS17 : PEReg<"RS17">, DwarfRegNum<[17]>;
def RS18 : PEReg<"RS18">, DwarfRegNum<[18]>;
def RS19 : PEReg<"RS19">, DwarfRegNum<[19]>;
def RS20 : PEReg<"RS20">, DwarfRegNum<[20]>;
def RS21 : PEReg<"RS21">, DwarfRegNum<[21]>;
def RS22 : PEReg<"RS22">, DwarfRegNum<[22]>;
def RS23 : PEReg<"RS23">, DwarfRegNum<[23]>;
def RS24 : PEReg<"RS24">, DwarfRegNum<[24]>;
def RS25 : PEReg<"RS25">, DwarfRegNum<[25]>;
def RS26 : PEReg<"RS26">, DwarfRegNum<[26]>;
def RS27 : PEReg<"RS27">, DwarfRegNum<[27]>;

// 邻接互连寄存器（E,S,W,N）
def RS28 : PEReg<"RS28">, DwarfRegNum<[28]>; // E
def RS29 : PEReg<"RS29">, DwarfRegNum<[29]>; // S
def RS30 : PEReg<"RS30">, DwarfRegNum<[30]>; // W
def RS31 : PEReg<"RS31">, DwarfRegNum<[31]>; // N

// 向量寄存器 VRS0 ~ VRS15，每个为8×i32
def VRS0  : PEReg<"VRS0">,  DwarfRegNum<[32]>;
def VRS1  : PEReg<"VRS1">,  DwarfRegNum<[33]>;
def VRS2  : PEReg<"VRS2">,  DwarfRegNum<[34]>;
def VRS3  : PEReg<"VRS3">,  DwarfRegNum<[35]>;
def VRS4  : PEReg<"VRS4">,  DwarfRegNum<[36]>;
def VRS5  : PEReg<"VRS5">,  DwarfRegNum<[37]>;
def VRS6  : PEReg<"VRS6">,  DwarfRegNum<[38]>;
def VRS7  : PEReg<"VRS7">,  DwarfRegNum<[39]>;
def VRS8  : PEReg<"VRS8">,  DwarfRegNum<[40]>;
def VRS9  : PEReg<"VRS9">,  DwarfRegNum<[41]>;
def VRS10 : PEReg<"VRS10">, DwarfRegNum<[42]>;
def VRS11 : PEReg<"VRS11">, DwarfRegNum<[43]>;
def VRS12 : PEReg<"VRS12">, DwarfRegNum<[44]>;
def VRS13 : PEReg<"VRS13">, DwarfRegNum<[45]>;
def VRS14 : PEReg<"VRS14">, DwarfRegNum<[46]>;
def VRS15 : PEReg<"VRS15">, DwarfRegNum<[47]>;

def SP : PEReg<"SP">, DwarfRegNum<[48]>;

// 标量寄存器类（包含邻接互连寄存器）
def GPR : RegisterClass<"PE", [i32], 32, (add
  SP,
  RS0,RS1,RS2,RS3,RS4,RS5,RS6,RS7,RS8,RS9,RS10,RS11,RS12,RS13,RS14,RS15,
  RS16,RS17,RS18,RS19,RS20,RS21,RS22,RS23,RS24,RS25,RS26,RS27,
  RS28,RS29,RS30,RS31
)>;

// 向量寄存器类（每个为8×i32）
def VR : RegisterClass<"PE", [v8i32], 256, (add
  VRS0,VRS1,VRS2,VRS3,VRS4,VRS5,VRS6,VRS7,
  VRS8,VRS9,VRS10,VRS11,VRS12,VRS13,VRS14,VRS15
)>;