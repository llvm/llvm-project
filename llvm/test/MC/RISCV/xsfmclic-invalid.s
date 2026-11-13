# RUN: llvm-mc --verify -triple riscv32 -mattr=-experimental-xsfmclic < %s

# RUN: llvm-mc --verify -triple riscv64 -mattr=-experimental-xsfmclic < %s

csrrs t1, sf.mtvt, zero // expected-error:11 {{system register 'sf.mtvt' requires 'experimental-xsfmclic' to be enabled}}

csrrs t1, sf.mnxti, zero // expected-error:11 {{system register 'sf.mnxti' requires 'experimental-xsfmclic' to be enabled}}

csrrs t1, sf.mintstatus, zero // expected-error:11 {{system register 'sf.mintstatus' requires 'experimental-xsfmclic' to be enabled}}

csrrs t1, sf.mscratchcsw, zero // expected-error:11 {{system register 'sf.mscratchcsw' requires 'experimental-xsfmclic' to be enabled}}

csrrs t1, sf.mscratchcswl, zero // expected-error:11 {{system register 'sf.mscratchcswl' requires 'experimental-xsfmclic' to be enabled}}
