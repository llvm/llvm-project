; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; Support of doubles is required.
; CHECK: OpCapability Float64
; CHECK: "fun01"
define spir_kernel void @fun01(i32 addrspace(1)* noalias %a, i32 addrspace(1)* %b, i32 %c) {
entry:
  %a.addr = alloca i32 addrspace(1)*, align 8
  %b.addr = alloca i32 addrspace(1)*, align 8
  %c.addr = alloca i32, align 4
  store i32 addrspace(1)* %a, i32 addrspace(1)** %a.addr, align 8
  store i32 addrspace(1)* %b, i32 addrspace(1)** %b.addr, align 8
  store i32 %c, i32* %c.addr, align 4
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %b.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 0
  %1 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %2 = load i32 addrspace(1)*, i32 addrspace(1)** %a.addr, align 8
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %2, i64 0
  store i32 %1, i32 addrspace(1)* %arrayidx1, align 4
  %3 = load i32 addrspace(1)*, i32 addrspace(1)** %b.addr, align 8
  %cmp = icmp ugt i32 addrspace(1)* %3, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %4 = load i32 addrspace(1)*, i32 addrspace(1)** %a.addr, align 8
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %4, i64 0
  store i32 2, i32 addrspace(1)* %arrayidx2, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; CHECK: "fun02"
define spir_kernel void @fun02(double addrspace(1)* %a, double addrspace(1)* %b, i32 %c) {
entry:
  %a.addr = alloca double addrspace(1)*, align 8
  %b.addr = alloca double addrspace(1)*, align 8
  %c.addr = alloca i32, align 4
  store double addrspace(1)* %a, double addrspace(1)** %a.addr, align 8
  store double addrspace(1)* %b, double addrspace(1)** %b.addr, align 8
  store i32 %c, i32* %c.addr, align 4
  %0 = load i32, i32* %c.addr, align 4
  %idxprom = sext i32 %0 to i64
  %1 = load double addrspace(1)*, double addrspace(1)** %b.addr, align 8
  %arrayidx = getelementptr inbounds double, double addrspace(1)* %1, i64 %idxprom
  %2 = load double, double addrspace(1)* %arrayidx, align 8
  %3 = load i32, i32* %c.addr, align 4
  %idxprom1 = sext i32 %3 to i64
  %4 = load double addrspace(1)*, double addrspace(1)** %a.addr, align 8
  %arrayidx2 = getelementptr inbounds double, double addrspace(1)* %4, i64 %idxprom1
  store double %2, double addrspace(1)* %arrayidx2, align 8
  ret void
}

; CHECK: "test_builtin"
define spir_func void @test_builtin(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %in.addr = alloca i32 addrspace(1)*, align 8
  %out.addr = alloca i32 addrspace(1)*, align 8
  %n = alloca i32, align 4
  store i32 addrspace(1)* %in, i32 addrspace(1)** %in.addr, align 8
  store i32 addrspace(1)* %out, i32 addrspace(1)** %out.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %conv = trunc i64 %call to i32
  store i32 %conv, i32* %n, align 4
  %0 = load i32, i32* %n, align 4
  %idxprom = sext i32 %0 to i64
  %1 = load i32 addrspace(1)*, i32 addrspace(1)** %in.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %1, i64 %idxprom
  %2 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = call spir_func i32 @_Z3absi(i32 %2)
  %3 = load i32, i32* %n, align 4
  %idxprom2 = sext i32 %3 to i64
  %4 = load i32 addrspace(1)*, i32 addrspace(1)** %out.addr, align 8
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %4, i64 %idxprom2
  store i32 %call1, i32 addrspace(1)* %arrayidx3, align 4
  ret void
}

; CHECK-NOT: "_Z13get_global_idj"
declare spir_func i64 @_Z13get_global_idj(i32)

; CHECK-NOT: "_Z3absi"
declare spir_func i32 @_Z3absi(i32)

; CHECK: "myabs"
define spir_func i32 @myabs(i32 %x) {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %call = call spir_func i32 @_Z3absi(i32 %0)
  ret i32 %call
}

; CHECK: "test_function_call"
define spir_func void @test_function_call(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %in.addr = alloca i32 addrspace(1)*, align 8
  %out.addr = alloca i32 addrspace(1)*, align 8
  %n = alloca i32, align 4
  store i32 addrspace(1)* %in, i32 addrspace(1)** %in.addr, align 8
  store i32 addrspace(1)* %out, i32 addrspace(1)** %out.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %conv = trunc i64 %call to i32
  store i32 %conv, i32* %n, align 4
  %0 = load i32, i32* %n, align 4
  %idxprom = sext i32 %0 to i64
  %1 = load i32 addrspace(1)*, i32 addrspace(1)** %in.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %1, i64 %idxprom
  %2 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = call spir_func i32 @myabs(i32 %2)
  %3 = load i32, i32* %n, align 4
  %idxprom2 = sext i32 %3 to i64
  %4 = load i32 addrspace(1)*, i32 addrspace(1)** %out.addr, align 8
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %4, i64 %idxprom2
  store i32 %call1, i32 addrspace(1)* %arrayidx3, align 4
  ret void
}
