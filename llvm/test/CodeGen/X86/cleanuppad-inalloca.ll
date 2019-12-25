; RUN: llc < %s | FileCheck %s

; Based on this C++:
; struct A {
;   int x;
;   A();
;   A(const A &a);
;   ~A();
; };
; extern "C" void takes_two(A a1, A a2);
; extern "C" void passes_two() { takes_two(A(), A()); }

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686--windows-msvc"

%struct.A = type { i32 }

define void @passes_two() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %argmem = alloca inalloca <{ %struct.A, %struct.A }>, align 4
  %0 = getelementptr inbounds <{ %struct.A, %struct.A }>, <{ %struct.A, %struct.A }>* %argmem, i32 0, i32 1
  %call = call x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"(%struct.A* %0)
  %1 = getelementptr inbounds <{ %struct.A, %struct.A }>, <{ %struct.A, %struct.A }>* %argmem, i32 0, i32 0
  %call1 = invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"(%struct.A* %1)
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  call void @takes_two(<{ %struct.A, %struct.A }>* inalloca nonnull %argmem)
  ret void

ehcleanup:                                        ; preds = %entry
  %2 = cleanuppad within none []
  call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %0) [ "funclet"(token %2) ]
  cleanupret from %2 unwind to caller
}

; CHECK: _passes_two:
; CHECK: pushl %ebp
; CHECK: movl %esp, %ebp
; CHECK: subl ${{[0-9]+}}, %esp
; CHECK: pushl %eax
; CHECK: pushl %eax
; CHECK: calll "??0A@@QAE@XZ"
; CHECK: calll "??0A@@QAE@XZ"
; CHECK: calll _takes_two
; 	ESP must be restored via EBP due to "dynamic" alloca.
; CHECK: leal -{{[0-9]+}}(%ebp), %esp
; CHECK: popl %ebp
; CHECK: retl

; CHECK: "?dtor$2@?0?passes_two@4HA":
; CHECK: pushl %ebp
; CHECK: subl $8, %esp
; CHECK: addl $12, %ebp
; CHECK: {{movl|leal}} -{{[0-9]+}}(%ebp), %ecx
; CHECK: calll "??1A@@QAE@XZ"
; CHECK: addl $8, %esp
; CHECK: retl

declare void @takes_two(<{ %struct.A, %struct.A }>* inalloca) #0

declare x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"(%struct.A* returned) #0

declare i32 @__CxxFrameHandler3(...)

declare x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A*) #0

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
