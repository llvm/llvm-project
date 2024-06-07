// RUN: mlir-opt %s -test-vector-scan-lowering -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @entry() {
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  %f3 = arith.constant 3.0: f32
  %f4 = arith.constant 4.0: f32
  %f5 = arith.constant 5.0: f32
  %f6 = arith.constant 6.0: f32

  // Construct test vector.
  %0 = vector.broadcast %f1 : f32 to vector<3x2xf32>
  %1 = vector.insert %f2, %0[0, 1] : f32 into vector<3x2xf32>
  %2 = vector.insert %f3, %1[1, 0] : f32 into vector<3x2xf32>
  %3 = vector.insert %f4, %2[1, 1] : f32 into vector<3x2xf32>
  %4 = vector.insert %f5, %3[2, 0] : f32 into vector<3x2xf32>
  %x = vector.insert %f6, %4[2, 1] : f32 into vector<3x2xf32>
  vector.print %x : vector<3x2xf32>
  // CHECK:  ( ( 1, 2 ), ( 3, 4 ), ( 5, 6 ) )

  %y = vector.broadcast %f6 : f32 to vector<2xf32>
  %z = vector.broadcast %f6 : f32 to vector<3xf32>
  // Scan
  %a:2 = vector.scan <add>, %x, %y {inclusive = true, reduction_dim = 0} :
    vector<3x2xf32>, vector<2xf32>
  %b:2 = vector.scan <add>, %x, %z {inclusive = true, reduction_dim = 1} :
    vector<3x2xf32>, vector<3xf32>
  %c:2 = vector.scan <add>, %x, %y {inclusive = false, reduction_dim = 0} :
    vector<3x2xf32>, vector<2xf32>
  %d:2 = vector.scan <add>, %x, %z {inclusive = false, reduction_dim = 1} :
    vector<3x2xf32>, vector<3xf32>

  // CHECK: ( ( 1, 2 ), ( 4, 6 ), ( 9, 12 ) )
  // CHECK: ( 9, 12 )
  // CHECK: ( ( 1, 3 ), ( 3, 7 ), ( 5, 11 ) )
  // CHECK: ( 3, 7, 11 )
  // CHECK: ( ( 6, 6 ), ( 7, 8 ), ( 10, 12 ) )
  // CHECK: ( 10, 12 )
  // CHECK: ( ( 6, 7 ), ( 6, 9 ), ( 6, 11 ) )
  // CHECK: ( 7, 9, 11 )
  vector.print %a#0 : vector<3x2xf32>
  vector.print %a#1 : vector<2xf32>
  vector.print %b#0 : vector<3x2xf32>
  vector.print %b#1 : vector<3xf32>
  vector.print %c#0 : vector<3x2xf32>
  vector.print %c#1 : vector<2xf32>
  vector.print %d#0 : vector<3x2xf32>
  vector.print %d#1 : vector<3xf32>

  return
}
