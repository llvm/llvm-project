// RUN: mlir-opt  %s -transform-interpreter  | FileCheck %s

  // CHECK: @optimize_shmem([[arg0:%.+]]: memref<{{.*}}>, [[readRow:%.+]]: index, [[readCol:%.+]]: index, [[writeRow:%.+]]: index, [[writeCol:%.+]]: index, [[fragRow:%.+]]: index, [[fragCol:%.+]]: index, [[fragColPerm:%.+]]: index, [[stRow:%.+]]: index, [[stCol:%.+]]: index)
  func.func @optimize_shmem(%arg0: memref<4096x4096xf16>, 
                    %readRow: index, %readCol: index,
                    %writeRow: index, %writeCol: index,
                    %fragRow: index, %fragCol: index, 
                    %fragColPerm: index,
                    %stRow: index, %stCol: index) {
    %cst = arith.constant 0.000000e+00 : f16

    %shmA = memref.alloc() {alignment = 64 : i64} : memref<128x32xf16, 3>
    %shmB = memref.alloc() {alignment = 64 : i64} : memref<256x32xf16, 3>

    %0 = vector.transfer_read %arg0[%readRow, %readCol], %cst {in_bounds = [true, true]} : memref<4096x4096xf16>, vector<1x8xf16>
    // CHECK: [[c7:%.+]] = arith.constant 7 : index                  
    // CHECK: [[srcBits:%.+]] = arith.andi [[stRow:%.+]], [[c7]]       
    // CHECK: [[c2:%.+]] = arith.constant 2 : index                 
    // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c2]]     
    // CHECK: [[stColPerm:%.+]] = arith.xori [[stCol:%.+]], [[xorBits]]  
    vector.transfer_write %0, %shmB[%writeRow, %writeCol] {in_bounds = [true, true]} : vector<1x8xf16>, memref<256x32xf16, 3>
    gpu.barrier
    gpu.barrier
    // CHECK: [[c7:%.+]] = arith.constant 7 : index                     
    // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c7]]     
    // CHECK: [[c2:%.+]] = arith.constant 2 : index                 
    // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c2]]       
    // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol:%.+]], [[xorBits]] 
    %1 = vector.load %shmB[%fragRow, %fragColPerm] : memref<256x32xf16, 3>, vector<8xf16>
    %2 = vector.transfer_read %arg0[%readRow, %readCol], %cst {in_bounds = [true, true]} : memref<4096x4096xf16>, vector<1x8xf16>
    // CHECK: [[c7:%.+]] = arith.constant 7 : index                  
    // CHECK: [[srcBits:%.+]] = arith.andi [[stRow:%.+]], [[c7]]       
    // CHECK: [[c2:%.+]] = arith.constant 2 : index                 
    // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c2]]     
    // CHECK: [[stColPerm:%.+]] = arith.xori [[stCol:%.+]], [[xorBits]]  
    vector.transfer_write %2, %shmA[%writeRow, %writeCol] {in_bounds = [true, true]} : vector<1x8xf16>, memref<128x32xf16, 3>
    gpu.barrier
    gpu.barrier
    // CHECK: [[c7:%.+]] = arith.constant 7 : index                     
    // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c7]]          
    // CHECK: [[c2:%.+]] = arith.constant 2 : index                     
    // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c2]] 
    // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol:%.+]], [[xorBits]]
    %3 = vector.load %shmA[%fragRow, %fragColPerm] : memref<128x32xf16, 3>, vector<8xf16>
    return
  }

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.amdgpu.optimize_shared_memory_reads_and_writes %0 : (!transform.any_op) -> ()
    transform.yield
  } // @__transform_main
} // module
