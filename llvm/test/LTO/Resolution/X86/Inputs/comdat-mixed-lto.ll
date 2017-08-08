; ModuleID = 'comdat-mixed-lto1.o'
source_filename = "comdat-mixed-lto1.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.Test::ptr" = type { i32 }

$C = comdat any

@C = linkonce_odr global %"class.Test::ptr" zeroinitializer, comdat, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @__cxx_global_var_init, i8* bitcast (%"class.Test::ptr"* @C to i8*) }]

define void @testglobfunc() #1 section ".text.startup" comdat($C) {
entry:
  ret void
}

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init() #1 section ".text.startup" comdat($C) {
entry:
  store i32 0, i32* getelementptr inbounds (%"class.Test::ptr", %"class.Test::ptr"* @C, i32 0, i32 0), align 4
  ret void
}
