// RUN: llvm-profdata merge %S/Inputs/regionMarkers.proftext -o %t.profdata

int main() {                      // CHECK: Marker at [[@LINE]]:12 = 1.11M
  int x = 0;

  if (x) {                        // CHECK: Marker at [[@LINE]]:10 = 0
    x = 0;
  } else {                        // CHECK: Marker at [[@LINE]]:10 = 1.11M
    x = 1;
  }
                                  // CHECK: Marker at [[@LINE+2]]:19 = 112M
                                  // CHECK: Marker at [[@LINE+1]]:28 = 111M
  for (int i = 0; i < 100; ++i) { // CHECK: Marker at [[@LINE]]:33 = 111M
    x = 1;
  }
                                  // CHECK: Marker at [[@LINE+1]]:16 = 1.11M
  x = x < 10 ? x + 1 : x - 1;     // CHECK: Marker at [[@LINE]]:24 = 0
  x = x > 10 ?
        x - 1:                    // CHECK: Marker at [[@LINE]]:9 = 0
        x + 1;                    // CHECK: Marker at [[@LINE]]:9 = 1.11M

  return 0;
}

// RUN: llvm-cov show %S/Inputs/regionMarkers.covmapping -instr-profile %t.profdata -show-regions -dump -filename-equivalence %s 2>&1 | FileCheck %s

// RUN: llvm-cov export %S/Inputs/regionMarkers.covmapping -instr-profile %t.profdata 2>&1 | FileCheck %S/Inputs/regionMarkers.json
