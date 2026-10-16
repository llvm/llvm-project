// RUN: llvm-mc -triple x86_64_lfi %s | FileCheck %s

stosq
// CHECK:      .bundle_lock
// CHECK-NEXT: movl %edi, %edi
// CHECK-NEXT: leaq (%r14,%rdi), %rdi
// CHECK-NEXT: stosq
// CHECK-NEXT: .bundle_unlock

stosb
// CHECK:      .bundle_lock
// CHECK-NEXT: movl %edi, %edi
// CHECK-NEXT: leaq (%r14,%rdi), %rdi
// CHECK-NEXT: stosb
// CHECK-NEXT: .bundle_unlock

movsq
// CHECK:      .bundle_lock
// CHECK-NEXT: movl %edi, %edi
// CHECK-NEXT: leaq (%r14,%rdi), %rdi
// CHECK-NEXT: movl %esi, %esi
// CHECK-NEXT: leaq (%r14,%rsi), %rsi
// CHECK-NEXT: movsq
// CHECK-NEXT: .bundle_unlock

cmpsl
// CHECK:      .bundle_lock
// CHECK-NEXT: movl %edi, %edi
// CHECK-NEXT: leaq (%r14,%rdi), %rdi
// CHECK-NEXT: movl %esi, %esi
// CHECK-NEXT: leaq (%r14,%rsi), %rsi
// CHECK-NEXT: cmpsl
// CHECK-NEXT: .bundle_unlock

scasb
// CHECK:      .bundle_lock
// CHECK-NEXT: movl %edi, %edi
// CHECK-NEXT: leaq (%r14,%rdi), %rdi
// CHECK-NEXT: scasb
// CHECK-NEXT: .bundle_unlock

lodsq
// CHECK:      .bundle_lock
// CHECK-NEXT: movl %esi, %esi
// CHECK-NEXT: leaq (%r14,%rsi), %rsi
// CHECK-NEXT: lodsq
// CHECK-NEXT: .bundle_unlock

rep stosq
// CHECK:      .bundle_lock
// CHECK-NEXT: movl %edi, %edi
// CHECK-NEXT: leaq (%r14,%rdi), %rdi
// CHECK-NEXT: rep {{.*}}stosq
// CHECK-NEXT: .bundle_unlock

repne cmpsb
// CHECK:      .bundle_lock
// CHECK-NEXT: movl %edi, %edi
// CHECK-NEXT: leaq (%r14,%rdi), %rdi
// CHECK-NEXT: movl %esi, %esi
// CHECK-NEXT: leaq (%r14,%rsi), %rsi
// CHECK-NEXT: repne {{.*}}cmpsb
// CHECK-NEXT: .bundle_unlock

rep
movsb
// CHECK:      .bundle_lock
// CHECK-NEXT: movl %edi, %edi
// CHECK-NEXT: leaq (%r14,%rdi), %rdi
// CHECK-NEXT: movl %esi, %esi
// CHECK-NEXT: leaq (%r14,%rsi), %rsi
// CHECK-NEXT: rep
// CHECK-NEXT: movsb
// CHECK-NEXT: .bundle_unlock
