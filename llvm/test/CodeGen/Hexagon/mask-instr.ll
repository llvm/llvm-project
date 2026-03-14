; Enable Utlilize mask instruction pass only on v66 and above.
; RUN: llc -mv60 -mtriple=hexagon < %s -o /dev/null

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@b = dso_local local_unnamed_addr global i8 0, align 1
@a = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: cold nounwind optsize memory(readwrite, argmem: none, inaccessiblemem: none)
define dso_local void @c() local_unnamed_addr {
entry:
  %0 = tail call i32 asm "", "=&r"()
  %and = and i32 %0, 134217727
  %tobool.not = icmp eq i32 %and, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %1 = load i8, ptr @b, align 1
  %loadedv = zext nneg i8 %1 to i32
  store i32 %loadedv, ptr @a, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}
