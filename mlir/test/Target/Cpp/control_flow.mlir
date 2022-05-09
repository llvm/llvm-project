// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s -check-prefix=CPP-DECLTOP

// simple(10, true)  -> 20
// simple(10, false) -> 30
func.func @simple(i64, i1) -> i64 {
^bb0(%a: i64, %cond: i1):
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%a: i64)
^bb2:
  %b = emitc.call "add"(%a, %a) : (i64, i64) -> i64
  cf.br ^bb3(%b: i64)
^bb3(%c: i64):
  cf.br ^bb4(%c, %a : i64, i64)
^bb4(%d : i64, %e : i64):
  %0 = emitc.call "add"(%d, %e) : (i64, i64) -> i64
  return %0 : i64
}
  // CPP-DECLTOP: int64_t simple(int64_t [[A:[^ ]*]], bool [[COND:[^ ]*]]) {
    // CPP-DECLTOP-NEXT: int64_t [[B:[^ ]*]];
    // CPP-DECLTOP-NEXT: int64_t [[V0:[^ ]*]];
    // CPP-DECLTOP-NEXT: int64_t [[C:[^ ]*]];
    // CPP-DECLTOP-NEXT: int64_t [[D:[^ ]*]];
    // CPP-DECLTOP-NEXT: int64_t [[E:[^ ]*]];
    // CPP-DECLTOP-NEXT: if ([[COND]]) {
    // CPP-DECLTOP-NEXT: goto [[BB1:[^ ]*]];
    // CPP-DECLTOP-NEXT: } else {
    // CPP-DECLTOP-NEXT: goto [[BB2:[^ ]*]];
    // CPP-DECLTOP-NEXT: }
    // CPP-DECLTOP-NEXT: [[BB1]]:
    // CPP-DECLTOP-NEXT: [[C]] = [[A]];
    // CPP-DECLTOP-NEXT: goto [[BB3:[^ ]*]];
    // CPP-DECLTOP-NEXT: [[BB2]]:
    // CPP-DECLTOP-NEXT: [[B]] = add([[A]], [[A]]);
    // CPP-DECLTOP-NEXT: [[C]] = [[B]];
    // CPP-DECLTOP-NEXT: goto [[BB3]];
    // CPP-DECLTOP-NEXT: [[BB3]]:
    // CPP-DECLTOP-NEXT: [[D]] = [[C]];
    // CPP-DECLTOP-NEXT: [[E]] = [[A]];
    // CPP-DECLTOP-NEXT: goto [[BB4:[^ ]*]];
    // CPP-DECLTOP-NEXT: [[BB4]]:
    // CPP-DECLTOP-NEXT: [[V0]] = add([[D]], [[E]]);
    // CPP-DECLTOP-NEXT: return [[V0]];


func.func @block_labels0() {
^bb1:
    cf.br ^bb2
^bb2:
    return
}
// CPP-DECLTOP: void block_labels0() {
  // CPP-DECLTOP-NEXT: goto label2;
  // CPP-DECLTOP-NEXT: label2:
  // CPP-DECLTOP-NEXT: return;
  // CPP-DECLTOP-NEXT: }


// Repeat the same function to make sure the names of the block labels get reset.
func.func @block_labels1() {
^bb1:
    cf.br ^bb2
^bb2:
    return
}
// CPP-DECLTOP: void block_labels1() {
  // CPP-DECLTOP-NEXT: goto label2;
  // CPP-DECLTOP-NEXT: label2:
  // CPP-DECLTOP-NEXT: return;
  // CPP-DECLTOP-NEXT: }
