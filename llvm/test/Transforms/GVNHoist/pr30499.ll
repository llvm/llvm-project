; RUN: opt -S -passes=gvn-hoist < %s

define void @_Z3fn2v() {
entry:
  %a = alloca ptr, align 8
  %b = alloca i32, align 4
  %0 = load ptr, ptr %a, align 8
  store i8 0, ptr %0, align 1
  %1 = load i32, ptr %b, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call i64 @_Z3fn1v()
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %b, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %2 = load ptr, ptr %a, align 8
  store i8 0, ptr %2, align 1
  ret void
}

; Function Attrs: nounwind readonly
declare i64 @_Z3fn1v()
