; RUN: opt -S -O2 < %s | FileCheck %s

; This test checks that -O2 is able to delete constructors that become empty
; only after some optimization passes have run, even if the pass structure
; changes.
; CHECK-NOT: @_GLOBAL__I_a

%class.Foo = type { i32 }

@foo = global %class.Foo zeroinitializer, align 4
@_ZN3Bar18LINKER_INITIALIZEDE = external constant i32
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_a, i8* null }]

define internal void @__cxx_global_var_init() section "__TEXT,__StaticInit,regular,pure_instructions" {
  %1 = load i32, i32* @_ZN3Bar18LINKER_INITIALIZEDE, align 4
  call void @_ZN3FooC1E17LinkerInitialized(%class.Foo* @foo, i32 %1)
  ret void
}

; Function Attrs: ssp uwtable
define linkonce_odr void @_ZN3FooC1E17LinkerInitialized(%class.Foo* %this, i32) unnamed_addr #0 align 2 {
  %2 = alloca %class.Foo*, align 8
  %3 = alloca i32, align 4
  store %class.Foo* %this, %class.Foo** %2, align 8
  store i32 %0, i32* %3, align 4
  %4 = load %class.Foo*, %class.Foo** %2
  %5 = load i32, i32* %3, align 4
  call void @_ZN3FooC2E17LinkerInitialized(%class.Foo* %4, i32 %5)
  ret void
}

; Function Attrs: nounwind ssp uwtable
define linkonce_odr void @_ZN3FooC2E17LinkerInitialized(%class.Foo* %this, i32) unnamed_addr #1 align 2 {
  %2 = alloca %class.Foo*, align 8
  %3 = alloca i32, align 4
  store %class.Foo* %this, %class.Foo** %2, align 8
  store i32 %0, i32* %3, align 4
  %4 = load %class.Foo*, %class.Foo** %2
  ret void
}

define internal void @_GLOBAL__I_a() section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @__cxx_global_var_init()
  ret void
}
