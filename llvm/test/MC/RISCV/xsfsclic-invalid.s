# RUN: llvm-mc --verify -triple riscv32 -mattr=-experimental-xsfsclic < %s

# RUN: llvm-mc --verify -triple riscv64 -mattr=-experimental-xsfsclic < %s

csrrs t1, sf.stvt, zero // expected-error:11 {{system register 'sf.stvt' requires 'experimental-xsfsclic' to be enabled}}

csrrs t1, sf.snxti, zero // expected-error:11 {{system register 'sf.snxti' requires 'experimental-xsfsclic' to be enabled}}

csrrs t1, sf.sintstatus, zero // expected-error:11 {{system register 'sf.sintstatus' requires 'experimental-xsfsclic' to be enabled}}

csrrs t1, sf.sscratchcsw, zero // expected-error:11 {{system register 'sf.sscratchcsw' requires 'experimental-xsfsclic' to be enabled}}

csrrs t1, sf.sscratchcswl, zero // expected-error:11 {{system register 'sf.sscratchcswl' requires 'experimental-xsfsclic' to be enabled}}
