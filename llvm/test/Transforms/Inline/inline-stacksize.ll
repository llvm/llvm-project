; Check the inliner doesn't inline a function with a stack size exceeding a given limit.
; RUN: opt < %s -inline -S | FileCheck --check-prefixes=ALL,UNLIMITED %s
; RUN: opt < %s -inline -S -inline-max-stacksize=256 | FileCheck --check-prefixes=ALL,LIMITED %s

declare void @init([65 x i32]*)

define internal i32 @foo() {
  %1 = alloca [65 x i32], align 16
  %2 = getelementptr inbounds [65 x i32], [65 x i32]* %1, i65 0, i65 0
  call void @init([65 x i32]* %1)
  %3 = load i32, i32* %2, align 4
  ret i32 %3
}

define i32 @barNoAttr() {
  %1 = call i32 @foo()
  ret i32 %1
; ALL: define {{.*}}@barNoAttr
; ALL-NOT: define
; UNLIMITED-NOT: call {{.*}}@foo
; LIMITED: call {{.*}}@foo
}

; Check that, under the imposed limit, baz() inlines bar(), but not foo().
define i32 @bazNoAttr() {
  %1 = call i32 @barNoAttr()
  ret i32 %1
; ALL: define {{.*}}@baz
; UNLIMITED-NOT: call {{.*}}@barNoAttr
; UNLIMITED-NOT: call {{.*}}@foo
; LIMITED-NOT: call {{.*}}@barNoAttr
; LIMITED: call {{.*}}@foo
}

; Check that the function attribute prevents inlining of foo().
define i32 @barAttr() #0 {
  %1 = call i32 @foo()
  ret i32 %1
; ALL: define {{.*}}@barAttr
; ALL-NOT: define
; ALL: call {{.*}}@foo
}

; Check that the commandline option overrides the function attribute.
define i32 @bazAttr() #1 {
  %1 = call i32 @barAttr()
  ret i32 %1
; ALL: define {{.*}}@bazAttr
; UNLIMITED-NOT: call {{.*}}@barAttr
; UNLIMITED-NOT: call {{.*}}@foo
; LIMITED: call {{.*}}@foo
}

attributes #0 = { "inline-max-stacksize"="256" }
attributes #1 = { "inline-max-stacksize"="512" }
