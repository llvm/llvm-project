# RUN: llvm-mc --verify=no-h -triple=riscv32 -mattr=+svinval < %s
# RUN: llvm-mc --verify=no-h -triple=riscv64 -mattr=+svinval < %s
# RUN: llvm-mc --verify=no-svinval -triple=riscv32 -mattr=+h < %s
# RUN: llvm-mc --verify=no-svinval -triple=riscv64 -mattr=+h < %s

hinval.vvma a0, a1
# no-h-error@-1:1 {{instruction requires the following: 'H' (Hypervisor)}}
# no-svinval-error@-2:1 {{instruction requires the following: 'Svinval' (Fine-Grained Address-Translation Cache Invalidation)}}

hinval.gvma a0, a1
# no-h-error@-1:1 {{instruction requires the following: 'H' (Hypervisor)}}
# no-svinval-error@-2:1 {{instruction requires the following: 'Svinval' (Fine-Grained Address-Translation Cache Invalidation)}}
