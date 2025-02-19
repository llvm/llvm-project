; RUN: opt -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 <  %s | FileCheck %s

declare { ptr, i1 } @get_struct()
declare <2 x ptr> @get_vec()
declare void @escape(ptr)

; CHECK: NoAlias: i32* %a, i32* %extract
define i32 @test_extractvalue() {
  %a = alloca i32
  %call = call { ptr, i1 } @get_struct()
  %extract = extractvalue { ptr, i1 } %call, 0
  store i32 0, ptr %extract
  %v = load i32, ptr %a
  ret i32 %v
}

; CHECK: NoAlias: i32* %a, i32* %extract
define i32 @test_extractelement() {
  %a = alloca i32
  %call = call <2 x ptr> @get_vec()
  %extract = extractelement <2 x ptr> %call, i32 0
  store i32 0, ptr %extract
  %v = load i32, ptr %a
  ret i32 %v
}

; CHECK: MayAlias: i32* %a, i32* %extract
define i32 @test_extractvalue_escape() {
  %a = alloca i32
  call void @escape(ptr %a)
  %call = call { ptr, i1 } @get_struct()
  %extract = extractvalue { ptr, i1 } %call, 0
  store i32 0, ptr %extract
  %v = load i32, ptr %a
  ret i32 %v
}

; CHECK: MayAlias: i32* %a, i32* %extract
define i32 @test_extractelement_escape() {
  %a = alloca i32
  call void @escape(ptr %a)
  %call = call <2 x ptr> @get_vec()
  %extract = extractelement <2 x ptr> %call, i32 0
  store i32 0, ptr %extract
  %v = load i32, ptr %a
  ret i32 %v
}
