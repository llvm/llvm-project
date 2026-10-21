class GenpayReg<bits<16> num, string n> : Register<n> {
  let HWEncoding = num;
  let Namespace = "Genpay";
}

foreach i = 0-14 in {
  def R#i : GenpayReg<i, "r"#i>;
}

// Stack pointer (not allocatable)
def SP : GenpayReg<15, "sp">;

def GPR : RegisterClass<"Genpay", [i64], 64, (add
  R0, R1, R2, R3, R4, R5, R6, R7,
  R8, R9, R10, R11, R12, R13, R14
)>;

def SPR : RegisterClass<"Genpay", [i64], 64, (add SP)>;
