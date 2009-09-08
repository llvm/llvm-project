; RUN: llvm-as < %s | llc -triple=x86_64-unknown-freebsd7.2 -code-model=kernel | FileCheck %s
; PR4689

%struct.__s = type { [8 x i8] }
%struct.pcb = type { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i16, i8* }
%struct.pcpu = type { i32*, i32*, i32*, i32*, %struct.pcb*, i64, i32, i32, i32, i32 }

define i64 @hammer_time(i64 %modulep, i64 %physfree) nounwind ssp noredzone noimplicitfloat {
; CHECK: hammer_time:
; CHECK: movq $Xrsvd, %rax
; CHECK: movq $Xrsvd, %rdi
; CHECK: movq $Xrsvd, %r8
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  br label %for.body

for.body:                                         ; preds = %for.inc, %if.end
  switch i32 undef, label %if.then76 [
    i32 9, label %for.inc
    i32 10, label %for.inc
    i32 11, label %for.inc
    i32 12, label %for.inc
  ]

if.then76:                                        ; preds = %for.body
  unreachable

for.inc:                                          ; preds = %for.body, %for.body, %for.body, %for.body
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc
  call void asm sideeffect "mov $1,%gs:$0", "=*m,r,~{dirflag},~{fpsr},~{flags}"(%struct.__s* bitcast (%struct.pcb** getelementptr (%struct.pcpu* null, i32 0, i32 4) to %struct.__s*), i64 undef) nounwind
  br label %for.body170

for.body170:                                      ; preds = %for.body170, %for.end
  store i64 or (i64 and (i64 or (i64 ptrtoint (void (i32, i32, i32, i32)* @Xrsvd to i64), i64 2097152), i64 2162687), i64 or (i64 or (i64 and (i64 shl (i64 ptrtoint (void (i32, i32, i32, i32)* @Xrsvd to i64), i64 32), i64 -281474976710656), i64 140737488355328), i64 15393162788864)), i64* undef
  br i1 undef, label %for.end175, label %for.body170

for.end175:                                       ; preds = %for.body170
  unreachable
}

declare void @Xrsvd(i32, i32, i32, i32) ssp noredzone noimplicitfloat
