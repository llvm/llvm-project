; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

define fastcc void @foo() {
        ret void
}

define coldcc void @bar() {
        call fastcc void @foo( )
        ret void
}

define void @structret({ i8 }* sret({ i8 })  %P) {
        call void @structret( { i8 }* sret({ i8 })  %P )
        ret void
}

define void @foo2() {
        ret void
}

define coldcc void @bar2() {
        call fastcc void @foo( )
        ret void
}

define cc42 void @bar3() personality i32 (...)* @__gxx_personality_v0 {
  invoke fastcc void @foo( )
    to label %Ok unwind label %U

Ok:
  ret void

U:
  %exn = landingpad {i8*, i32}
            cleanup
  resume { i8*, i32 } %exn
}

define void @bar4() personality i32 (...)* @__gxx_personality_v0 {
  call cc42 void @bar( )
  invoke cc42 void @bar3( )
    to label %Ok unwind label %U

Ok:
  ret void

U:
  %exn = landingpad {i8*, i32}
            cleanup
  resume { i8*, i32 } %exn
}

declare ghccc void @ghc_callee()

define void @ghc_caller() {
  call ghccc void @ghc_callee()
  ret void
}

declare hhvm_ccc void @hhvm_c_callee()

define hhvmcc void @hhvm_caller() {
  call hhvm_ccc void @hhvm_c_callee()
  ret void
}

declare i32 @__gxx_personality_v0(...)
