; RUN: llc < %s -mtriple=i386-pc-win32 -mattr=+sse | FileCheck --check-prefix=WIN32 %s
; RUN: llc < %s -mtriple=x86_64-win32 -mattr=+sse | FileCheck --check-prefix=WIN64 %s
; RUN: llc < %s -mtriple=x86_64-linux-gnu -mattr=+sse | FileCheck --check-prefix=LINUXOSX %s

; WIN32-LABEL:  test_argReti1:
; WIN32:        incb  %al
; WIN32:        ret{{.*}}

; WIN64-LABEL:  test_argReti1:
; WIN64:        incb  %al
; WIN64:        ret{{.*}}

; Test regcall when receiving/returning i1
define x86_regcallcc i1 @test_argReti1(i1 %a)  {
  %add = add i1 %a, 1
  ret i1 %add
}

; WIN32-LABEL:  test_CallargReti1:
; WIN32:        movzbl  %al, %eax
; WIN32:        call{{.*}}   {{.*}}test_argReti1
; WIN32:        incb  %al
; WIN32:        ret{{.*}}

; WIN64-LABEL:  test_CallargReti1:
; WIN64:        movzbl  %al, %eax
; WIN64:        call{{.*}}   {{.*}}test_argReti1
; WIN64:        incb  %al
; WIN64:        ret{{.*}}

; Test regcall when passing/retrieving i1
define x86_regcallcc i1 @test_CallargReti1(i1 %a)  {
  %b = add i1 %a, 1
  %c = call x86_regcallcc i1 @test_argReti1(i1 %b)
  %d = add i1 %c, 1
  ret i1 %d
}

; WIN64-LABEL: testf32_inp
; WIN64: movaps {{%xmm(1[2-5])}}, {{.*(%r(b|s)p).*}}  {{#+}} 16-byte Spill
; WIN64: movaps {{%xmm(1[2-5])}}, {{.*(%r(b|s)p).*}}  {{#+}} 16-byte Spill
; WIN64: movaps {{%xmm(1[2-5])}}, {{.*(%r(b|s)p).*}}  {{#+}} 16-byte Spill
; WIN64: movaps {{%xmm(1[2-5])}}, {{.*(%r(b|s)p).*}}  {{#+}} 16-byte Spill
; WIN64: {{.*}} {{%xmm([0-9]|1[0-1])}}, {{%xmm(1[2-5])}}
; WIN64: {{.*}} {{%xmm([0-9]|1[0-1])}}, {{%xmm(1[2-5])}}
; WIN64: {{.*}} {{%xmm([0-9]|1[0-1])}}, {{%xmm(1[2-5])}}
; WIN64: {{.*}} {{%xmm([0-9]|1[0-1])}}, {{%xmm(1[2-5])}}
; WIN64: movaps {{.*(%r(b|s)p).*}}, {{%xmm(1[2-5])}}  {{#+}} 16-byte Reload
; WIN64: movaps {{.*(%r(b|s)p).*}}, {{%xmm(1[2-5])}}  {{#+}} 16-byte Reload
; WIN64: movaps {{.*(%r(b|s)p).*}}, {{%xmm(1[2-5])}}  {{#+}} 16-byte Reload
; WIN64: movaps {{.*(%r(b|s)p).*}}, {{%xmm(1[2-5])}}  {{#+}} 16-byte Reload
; WIN64: retq

; WIN32-LABEL: testf32_inp
; WIN32: movaps {{%xmm([0-7])}}, {{.*(%e(b|s)p).*}}  {{#+}} 16-byte Spill
; WIN32: {{.*}} {{%xmm[0-7]}}, {{%xmm[4-7]}}
; WIN32: {{.*}} {{%xmm[0-7]}}, {{%xmm[4-7]}}
; WIN32: {{.*}} {{%xmm[0-7]}}, {{%xmm[4-7]}}
; WIN32: {{.*}} {{%xmm[0-7]}}, {{%xmm[4-7]}}
; WIN32: movaps {{.*(%e(b|s)p).*}}, {{%xmm([0-7])}}  {{#+}} 16-byte Reload
; WIN32: retl

; LINUXOSX-LABEL: testf32_inp
; LINUXOSX: movaps {{%xmm(1[2-5])}}, {{.*(%r(b|s)p).*}}  {{#+}} 16-byte Spill
; LINUXOSX: movaps {{%xmm(1[2-5])}}, {{.*(%r(b|s)p).*}}  {{#+}} 16-byte Spill
; LINUXOSX: movaps {{%xmm(1[2-5])}}, {{.*(%r(b|s)p).*}}  {{#+}} 16-byte Spill
; LINUXOSX: movaps {{%xmm(1[2-5])}}, {{.*(%r(b|s)p).*}}  {{#+}} 16-byte Spill
; LINUXOSX: {{.*}} {{%xmm([0-9]|1[0-1])}}, {{%xmm(1[2-5])}}
; LINUXOSX: {{.*}} {{%xmm([0-9]|1[0-1])}}, {{%xmm(1[2-5])}}
; LINUXOSX: {{.*}} {{%xmm([0-9]|1[0-1])}}, {{%xmm(1[2-5])}}
; LINUXOSX: {{.*}} {{%xmm([0-9]|1[0-1])}}, {{%xmm(1[2-5])}}
; LINUXOSX: movaps {{.*(%r(b|s)p).*}}, {{%xmm(1[2-5])}}  {{#+}} 16-byte Reload
; LINUXOSX: movaps {{.*(%r(b|s)p).*}}, {{%xmm(1[2-5])}}  {{#+}} 16-byte Reload
; LINUXOSX: movaps {{.*(%r(b|s)p).*}}, {{%xmm(1[2-5])}}  {{#+}} 16-byte Reload
; LINUXOSX: movaps {{.*(%r(b|s)p).*}}, {{%xmm(1[2-5])}}  {{#+}} 16-byte Reload
; LINUXOSX: retq

;test calling conventions - input parameters, callee saved XMMs
define x86_regcallcc <16 x float> @testf32_inp(<16 x float> %a, <16 x float> %b, <16 x float> %c) nounwind {
  %x1 = fadd <16 x float> %a, %b
  %x2 = fmul <16 x float> %a, %b
  %x3 = fsub <16 x float> %x1, %x2
  %x4 = fadd <16 x float> %x3, %c
  ret <16 x float> %x4
}

; WIN32-LABEL: testi32_inp
; WIN32: pushl {{%e(si|di|bx|bp)}}
; WIN32: pushl {{%e(si|di|bx|bp)}}
; WIN32: popl {{%e(si|di|bx|bp)}}
; WIN32: popl {{%e(si|di|bx|bp)}}
; WIN32: retl

; WIN64-LABEL: testi32_inp
; WIN64: pushq	{{%r(bp|bx|1[0-5])}}
; WIN64: pushq	{{%r(bp|bx|1[0-5])}}
; WIN64: pushq	{{%r(bp|bx|1[0-5])}}
; WIN64: popq	{{%r(bp|bx|1[0-5])}}
; WIN64: popq	{{%r(bp|bx|1[0-5])}}
; WIN64: popq	{{%r(bp|bx|1[0-5])}}
; WIN64: retq

; LINUXOSX-LABEL: testi32_inp
; LINUXOSX: pushq	{{%r(bp|bx|1[2-5])}}
; LINUXOSX: pushq	{{%r(bp|bx|1[2-5])}}
; LINUXOSX: popq	{{%r(bp|bx|1[2-5])}}
; LINUXOSX: popq	{{%r(bp|bx|1[2-5])}}
; LINUXOSX: retq

;test calling conventions - input parameters, callee saved GPRs
define x86_regcallcc i32 @testi32_inp(i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6,
                                      i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6) nounwind {
  %x1 = sub i32 %a1, %a2
  %x2 = sub i32 %a3, %a4
  %x3 = sub i32 %a5, %a6
  %y1 = sub i32 %b1, %b2
  %y2 = sub i32 %b3, %b4
  %y3 = sub i32 %b5, %b6
  %v1 = add i32 %a1, %a2
  %v2 = add i32 %a3, %a4
  %v3 = add i32 %a5, %a6
  %w1 = add i32 %b1, %b2
  %w2 = add i32 %b3, %b4
  %w3 = add i32 %b5, %b6
  %s1 = mul i32 %x1, %y1
  %s2 = mul i32 %x2, %y2
  %s3 = mul i32 %x3, %y3
  %t1 = mul i32 %v1, %w1
  %t2 = mul i32 %v2, %w2
  %t3 = mul i32 %v3, %w3
  %m1 = add i32 %s1, %s2
  %m2 = add i32 %m1, %s3
  %n1 = add i32 %t1, %t2
  %n2 = add i32 %n1, %t3
  %r1 = add i32 %m2, %n2
  ret i32 %r1
}

; X32: testf32_stack
; X32: movaps {{%xmm([0-7])}}, {{(-*[0-9])+}}(%ebp)
; X32: movaps {{%xmm([0-7])}}, {{(-*[0-9])+}}(%ebp)
; X32: movaps {{%xmm([0-7])}}, {{(-*[0-9])+}}(%ebp)
; X32: movaps {{%xmm([0-7])}}, {{(-*[0-9])+}}(%ebp)
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: addps {{([0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: movaps {{(-*[0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: movaps {{(-*[0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: movaps {{(-*[0-9])+}}(%ebp), {{%xmm([0-7])}}
; X32: movaps {{(-*[0-9])+}}(%ebp), {{%xmm([0-7])}}

; LINUXOSX: testf32_stack
; LINUXOSX: addps {{%xmm([0-9]+)}}, {{%xmm([0-9]+)}}
; LINUXOSX: addps {{%xmm([0-9]+)}}, {{%xmm([0-9]+)}}
; LINUXOSX: addps {{%xmm([0-9]+)}}, {{%xmm([0-9]+)}}
; LINUXOSX: addps {{%xmm([0-9]+)}}, {{%xmm([0-9]+)}}
; LINUXOSX: addps {{%xmm([0-9]+)}}, {{%xmm([0-9]+)}}
; LINUXOSX: addps {{%xmm([0-9]+)}}, {{%xmm([0-9]+)}}
; LINUXOSX: addps {{%xmm([0-9]+)}}, {{%xmm([0-9]+)}}
; LINUXOSX: addps {{%xmm([0-9]+)}}, {{%xmm([0-9]+)}}
; LINUXOSX: addps {{([0-9])+}}(%rsp), {{%xmm([0-7])}}
; LINUXOSX: addps {{([0-9])+}}(%rsp), {{%xmm([0-7])}}
; LINUXOSX: addps {{([0-9])+}}(%rsp), {{%xmm([0-7])}}
; LINUXOSX: addps {{([0-9])+}}(%rsp), {{%xmm([0-7])}}
; LINUXOSX: addps {{([0-9])+}}(%rsp), {{%xmm([0-7])}}
; LINUXOSX: addps {{([0-9])+}}(%rsp), {{%xmm([0-7])}}
; LINUXOSX: addps {{([0-9])+}}(%rsp), {{%xmm([0-7])}}
; LINUXOSX: addps {{([0-9])+}}(%rsp), {{%xmm([0-7])}}
; LINUXOSX: retq

; Test that parameters, overflowing register capacity, are passed through the stack
define x86_regcallcc <32 x float> @testf32_stack(<32 x float> %a, <32 x float> %b, <32 x float> %c) nounwind {
  %x1 = fadd <32 x float> %a, %b
  %x2 = fadd <32 x float> %x1, %c
  ret <32 x float> %x2
}
