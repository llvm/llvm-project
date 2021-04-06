; RUN:  opt < %s -print-callgraph -disable-output 2>&1 | FileCheck %s

; CHECK: Call graph node <<null function>><<{{.*}}>>  #uses=0
; CHECK-NEXT:   CS<None> calls function 'foo'
; CHECK-EMPTY:
; CHECK-NEXT: Call graph node for function: 'bar'<<{{.*}}>>  #uses=1
; CHECK-EMPTY:
; CHECK-NEXT: Call graph node for function: 'foo'<<{{.*}}>>  #uses=1
; CHECK-EMPTY:
; CHECK-NEXT: Call graph node for function: 'test'<<{{.*}}>>  #uses=0
; CHECK-NEXT:   CS<{{.*}}> calls function 'bar'

define internal i32 @foo() {
entry:
    ret i32 5
}

define internal i32 @bar(float()* %arg) {
    ret i32 5
}

define internal i32 @test() {
  %v1 = call i32 @bar(float()* bitcast (i32()* @foo to float()*))
  %v2 = add i32 %v1, 6
  ret i32 %v2
}

