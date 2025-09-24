// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not llvm-tblgen -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// RUN: not llvm-tblgen -DERROR4 %s 2>&1 | FileCheck --check-prefix=ERROR4 %s
// RUN: not llvm-tblgen -DERROR5 %s 2>&1 | FileCheck --check-prefix=ERROR5 %s
// RUN: not llvm-tblgen -DERROR6 %s 2>&1 | FileCheck --check-prefix=ERROR6 %s
// RUN: not llvm-tblgen -DERROR7 %s 2>&1 | FileCheck --check-prefix=ERROR7 %s
// XFAIL: vg_leak

// CHECK: def shifts
// CHECK: shifted_b = 8
// CHECK: shifted_i = 8
def shifts {
    bits<2> b = 0b10;
    int i = 2;
    int shifted_b = !shl(b, 2);
    int shifted_i = !shl(i, 2);
}

class Int<int value> {
  int Value = value;
}

def int2 : Int<2>;
def int1022 : Int<1022>;
def int1024 : Int<1024>;

// CHECK: def v0a
// CHECK: Value = 0
def v0a : Int<!sub(int1024.Value, int1024.Value)>;

// CHECK: def v0b
// CHECK: Value = 0
def v0b : Int<!and(int1024.Value, 2048)>;

// CHECK: def v1
// CHECK: Value = 1
def v1 : Int<!and(1025, 1)>;

// CHECK: def v1019
// CHECK: Value = 1019
def v1019 : Int<!sub(int1022.Value, 3)>;

// CHECK: def v1023
// CHECK: Value = 1023
def v1023 : Int<!or(int1022.Value, 1)>;

def v1025   : Int<!add(int1024.Value, 1)>;
// CHECK: def v1025
// CHECK: Value = 1025

// CHECK: def v12
// CHECK: Value = 12
def v12   : Int<!mul(4, 3)>;

// CHECK: def v13
// CHECK: Value = 5
def v13   : Int<!div(10, 2)>;

// CHECK: def v14
// CHECK: Value = 5
def v14 : Int<!div(11, 2)>;

// CHECK: def v15
// CHECK: Value = 1
def v15 : Int<!div(1, 1)>;

// CHECK: def v16
// CHECK: Value = 0
def v16 : Int<!div(0, 10)>;

// CHECK: def v17
// CHECK: Value = -2
def v17 : Int<!div(-8, 4)>;

#ifdef ERROR1
// ERROR1: error: Illegal operation: division by zero
def v18 : Int<!div(4, 0)>;
#endif

#ifdef ERROR2
// ERROR2: error: Illegal operation: INT64_MIN / -1
def v19 : Int<!div(-9223372036854775808, -1)>;
#endif

// CHECK: def v1a
// CHECK: Value = 1

// CHECK: def v2048
// CHECK: Value = 2048
def v2048   : Int<!add(int1024.Value, int1024.Value)>;

// CHECK: def v3072
// CHECK: Value = 3072
def v3072 : Int<!or(int1024.Value, v2048.Value)>;

// CHECK: def v4
// CHECK: Value = 4

// CHECK: def v7
// CHECK: Value = 7

def v4 : Int<!add(int2.Value, 1, v1.Value)>;
def v7 : Int<!or(v1.Value, int2.Value, v4.Value)>;
def v1a : Int<!and(v7.Value, 5, v1.Value)>;

// CHECK: def v84
// CHECK: Value = 84
def v84   : Int<!mul(v12.Value, v7.Value)>;

// CHECK: def v9
// CHECK: Value = 9
def v9 : Int<!xor(v7.Value, 0x0E)>;

// CHECK: def v924
// CHECK: Value = 924
def v924   : Int<!mul(v84.Value, 11)>;

// CHECK: def v925
// CHECK: Value = 925
def v925 : Int<!sub(v924.Value, -1)>;

// CHECK: def v950
// CHECK: Value = 4
def v950: Int<!logtwo(16)>;

// CHECK: def v951
// CHECK: Value = 10
def v951 : Int<!logtwo(1024)>;

// CHECK: def v952
// CHECK: Value = 10
def v952 : Int<!logtwo(1025)>;

#ifdef ERROR3
// ERROR3: error: Illegal operation: logtwo is undefined on arguments less than or equal to 0
def v953 : Int<!logtwo(0)>;
#endif

#ifdef ERROR4
// ERROR4: error: Illegal operation: logtwo is undefined on arguments less than or equal to 0
def v954 : Int<!logtwo(-1)>;
#endif

// CHECK: def vneg
// CHECK: Value = -2
def vneg : Int<!sub(v925.Value, 927)>;

#ifdef ERROR5
// ERROR5: error: Illegal operation: out of bounds shift
def vshl_neg : Int<!shl(1, -1)>;
#endif

#ifdef ERROR6
// ERROR6: error: Illegal operation: out of bounds shift
def vsra_large : Int<!sra(1, 64)>;
#endif

#ifdef ERROR7
// ERROR7: error: Illegal operation: out of bounds shift
def vsrl_large : Int<!srl(1, 100)>;
#endif
