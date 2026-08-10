// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/showTabsHTML.proftext
// RUN: llvm-cov show %S/Inputs/showTabsHTML.covmapping -format html -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck %s --strict-whitespace

int main(int argc, char ** argv) {
	(void) "This tab starts at column 0";              // CHECK: >  (void) &quot;This tab starts at column 0&quot;;
  (void) "	This tab starts at column 10";           // CHECK: >  (void) &quot;  This tab starts at column 10&quot;;
  (void) "This 	 tab starts at column 15";           // CHECK: >  (void) &quot;This   tab starts at column 15&quot;;

  return 0;
}

// RUN: llvm-cov show %S/Inputs/showTabsHTML.covmapping -format html -tab-size=3 -instr-profile %t.profdata -path-equivalence=/tmp,%S %s | FileCheck -check-prefix=CHECK-TABSIZE %s

// CHECK-TABSIZE: >  (void) &quot;This tab starts at column 0&quot;;
// CHECK-TABSIZE: >  (void) &quot;   This tab starts at column 10&quot;;
// CHECK-TABSIZE: >  (void) &quot;This    tab starts at column 15&quot;;
