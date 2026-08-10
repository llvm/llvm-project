// RUN: llvm-tblgen %s | FileCheck %s

class PrintString<string s> {
 string S = s;
}

def op;
def case_01 : PrintString<!repr((op op, op))>;
// CHECK-LABEL: def case_01 { // PrintString
// CHECK-NEXT:    string S = "(op op, op)";
// CHECK-NEXT:  }

def case_02 : PrintString<!repr(32)>;
// CHECK-LABEL: def case_02 { // PrintString
// CHECK-NEXT:    string S = "32";
// CHECK-NEXT:  }

multiclass Multi<int i, bits<3> b> {
  defvar IncI = !repr(!add(i,1));
  def _IncI : PrintString<IncI>;
  def _b : PrintString<!repr(b)>;
}

defm case_03 : Multi<2, 0b111>;
// CHECK-LABEL: def case_03_IncI {      // PrintString
// CHECK-NEXT:    string S = "3";
// CHECK-NEXT:  }
// CHECK-LABEL: def case_03_b { // PrintString
// CHECK-NEXT:    string S = "{ 1, 1, 1 }";
// CHECK-NEXT:  }

def case_04 : PrintString<!repr(!foreach(i, [1,2,3], !mul(i,2)))>;
// CHECK-LABEL: def case_04 { // PrintString
// CHECK-NEXT:    string S = "[2, 4, 6]";
// CHECK-NEXT:  }

// Show the dumping of full records...
def case_05 : PrintString<!repr(case_04)>;
// CHECK-LABEL: def case_05 { // PrintString
// CHECK-NEXT:    string S = "case_04 { // PrintString
// CHECK-NEXT:    string S = "[2, 4, 6]";
// CHECK-NEXT:  }
// CHECK-NEXT:  ";
// CHECK-NEXT:  }

// ... and how !repr differs compared to !cast<string>.
def case_06 : PrintString<!cast<string>(case_04)>;
// CHECK-LABEL: def case_06 { // PrintString
// CHECK-NEXT:    string S = "case_04";
// CHECK-NEXT:  }

