; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo --opaque-pointers > %t.echo
; RUN: diff -w %t.orig %t.echo

%S = type { i32, i32 }

define i32 @method(ptr %this, i32 %arg.a, i32 %arg.b) {
  %a = alloca i32, align 4
  store i32 %arg.a, ptr %a, align 4
  %b = alloca i32, align 4
  store i32 %arg.b, ptr %b
  %1 = load i32, ptr %a, align 4
  %2 = load i32, ptr %b, align 4
  %3 = add i32 %1, %2
  %4 = getelementptr inbounds %S, ptr %this, i32 0, i32 0
  %5 = load i32, ptr %4, align 4
  %6 = add i32 %3, %5
  %7 = getelementptr inbounds %S, ptr %this, i32 0, i32 1
  %8 = load i32, ptr %7, align 4
  %9 = add i32 %6, %8
  ret i32 %9
}

define i32 @main() {
  %s = alloca %S, align 4
  store %S zeroinitializer, ptr %s
  %1 = getelementptr inbounds %S, ptr %s, i32 0, i32 0
  %2 = getelementptr inbounds %S, ptr %s, i32 0, i32 1
  store i32 1, ptr %2
  store i32 1, ptr %1
  %3 = insertvalue { ptr, ptr } undef, ptr %s, 0
  %4 = insertvalue { ptr, ptr } %3, ptr @method, 1
  %5 = extractvalue { ptr, ptr } %4, 0
  %6 = extractvalue { ptr, ptr } %4, 1
  %7 = call i32 %6(ptr %5, i32 38, i32 2)
  ret i32 %7
}
