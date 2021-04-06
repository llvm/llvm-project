; RUN:  opt < %s -print-callgraph -disable-output 2>&1 | FileCheck %s
; CHECK: Call graph node <<null function>><<{{.*}}>>  #uses=0
; CHECK-EMPTY:
; CHECK-NEXT: Call graph node for function: 'foo'<<{{.*}}>>  #uses=0
; CHECK-EMPTY:
; CHECK-NEXT: Call graph node for function: 'test_bitcast_callees2'<<{{.*}}>>  #uses=0
; CHECK-NEXT:   CS<{{.*}}> calls external node
; CHECK-NEXT:   CS<{{.*}}> calls external node
; CHECK-NEXT:   CS<{{.*}}> calls external node

define internal i32 @foo() {
entry:
    ret i32 5
}

define internal float @test_bitcast_callees2() {
  %v1 = call float bitcast (i32()* @foo to float()*)()
  %v2 = fadd float %v1, 1.0
  %v3 = call float bitcast (i32()* @foo to float()*)()
  %v4 = fadd float %v3, %v2
  %v5 = call i8 bitcast (i32()* @foo to i8()*)()
  %v6 = uitofp i8 %v5 to float
  %v7 = fadd float %v4, %v6
  ret float %v7
}

