; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-DEFAULT
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-emit-op-names %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-ALL-NAMES
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Verify that OpName is generated for Global Variables, Functions, Parameters,
; Local Variables (Alloca), and Basic Blocks (Labels).
; We preserve these names because they significantly improve the readability of
; the generated SPIR-V binary and are unlikely to inhibit optimizations (like
; Dead Code Elimination) since they define the interface or storage of the program.

; 1. Global variables ("GlobalVar")
; CHECK-DAG: OpName %[[#GlobalVar:]] "GlobalVar"

; 2. Functions ("test_names")
; CHECK-DAG: OpName %[[#Func:]] "test_names"

; 3. Function parameters ("param")
; CHECK-DAG: OpName %[[#Param:]] "param"

; 4. Local variables (AllocaInst) ("localVar")
; CHECK-DAG: OpName %[[#LocalVar:]] "localVar"

; 5. Basic Blocks ("entry", "body")
; CHECK-DAG: OpName %[[#Entry:]] "entry"
; CHECK-DAG: OpName %[[#Body:]] "body"

; Verify that OpName is NOT generated for intermediate instructions
; (arithmetic, etc.) by default. This reduces file size and noise, and prevents
; potential interference with optimizations.
; With --spirv-emit-op-names, we expect them to be generated.

; CHECK-DEFAULT-NOT: OpName %{{.*}} "add"
; CHECK-DEFAULT-NOT: OpName %{{.*}} "sub"

; CHECK-ALL-NAMES-DAG: OpName %{{.*}} "add"
; CHECK-ALL-NAMES-DAG: OpName %{{.*}} "sub"

@GlobalVar = global i32 0

define spir_func void @test_names(i32 %param) {
entry:
  %localVar = alloca i32
  br label %body

body:
  %add = add i32 %param, 1
  %sub = sub i32 %add, 1
  store i32 %sub, ptr %localVar
  ret void
}
