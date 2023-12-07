; RUN: llc -march=mips -mcpu=mips32r6 < %s | FileCheck %s --check-prefixes=ALL,R6
; RUN: llc -march=mips -mcpu=mips64r6 -target-abi=n64 -relocation-model=pic \
; RUN:     < %s | FileCheck %s --check-prefixes=ALL,R6
; RUN: llc -march=mips -mcpu=mips32 < %s | FileCheck %s --check-prefixes=ALL,PRER6
; RUN: llc -march=mips -mcpu=mips64 -target-abi=n64 -relocation-model=pic \
; RUN:     < %s | FileCheck %s --check-prefixes=ALL,PRER6


%struct.anon = type { [63 x i32], i32, i32 }

define i32 @Atomic() {
; CHECK-LABEL: Atomic:
entry:
  %s = alloca %struct.anon, align 4
  %count = getelementptr inbounds %struct.anon, ptr %s, i64 0, i32 1
  store i32 0, ptr %count, align 4
; R6: addiu $[[R0:[0-9a-z]+]], $sp, {{[0-9]+}}

; ALL: #APP

; R6: ll ${{[0-9a-z]+}}, 0($[[R0]])
; R6: sc ${{[0-9a-z]+}}, 0($[[R0]])

; PRER6: ll ${{[0-9a-z]+}}, {{[0-9]+}}(${{[0-9a-z]+}})
; PRER6: sc ${{[0-9a-z]+}}, {{[0-9]+}}(${{[0-9a-z]+}})

; ALL: #NO_APP

  %0 = call { i32, i32 } asm sideeffect ".set push\0A.set noreorder\0A1:\0All $0, $2\0Aaddu $1, $0, $3\0Asc $1, $2\0Abeqz $1, 1b\0Aaddu $1, $0, $3\0A.set pop\0A", "=&r,=&r,=*^ZC,Ir,*^ZC,~{memory},~{$1}"(ptr elementtype(i32) %count, i32 10, ptr elementtype(i32) %count)
  %asmresult1.i = extractvalue { i32, i32 } %0, 1
  %cmp = icmp ne i32 %asmresult1.i, 10
  %conv = zext i1 %cmp to i32
  %call2 = call i32 @f(i32 signext %conv)
  ret i32 %call2
}

declare i32 @f(i32 signext)
