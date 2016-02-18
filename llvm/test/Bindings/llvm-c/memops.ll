; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo

%S = type { i32, i32 }

define i32 @method(%S* %this, i32 %arg.a, i32 %arg.b) {
  %a = alloca i32
  store i32 %arg.a, i32* %a, align 4
  %b = alloca i32
  store i32 %arg.b, i32* %b
  %1 = load i32, i32* %a, align 4
  %2 = load i32, i32* %b, align 4
  %3 = add i32 %1, %2
  %4 = getelementptr inbounds %S, %S* %this, i32 0, i32 0
  %5 = load i32, i32* %4, align 4
  %6 = add i32 %3, %5
  %7 = getelementptr inbounds %S, %S* %this, i32 0, i32 1
  %8 = load i32, i32* %7, align 4
  %9 = add i32 %6, %8
  ret i32 %9
}

define i32 @main() {
  %s = alloca %S
  store %S zeroinitializer, %S* %s
  %1 = getelementptr inbounds %S, %S* %s, i32 0, i32 0
  %2 = getelementptr inbounds %S, %S* %s, i32 0, i32 1
  store i32 1, i32* %2
  store i32 1, i32* %1
  %3 = insertvalue { %S*, i32 (%S*, i32, i32)* } undef, %S* %s, 0
  %4 = insertvalue { %S*, i32 (%S*, i32, i32)* } %3, i32 (%S*, i32, i32)* @method, 1
  %5 = extractvalue { %S*, i32 (%S*, i32, i32)* } %4, 0
  %6 = extractvalue { %S*, i32 (%S*, i32, i32)* } %4, 1
  %7 = call i32 %6(%S* %5, i32 38, i32 2)
  ret i32 %7
}
