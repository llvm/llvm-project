# Instructions that are invalid and are correctly rejected but used to emit
# the wrong error message.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips1 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        ldc2      $8,-21181($at)  # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        ldc2      $8,-1024($at)   # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        ldc3      $29,-28645($s1) # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        ll        $v0,-7321($s2)  # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        sc        $t7,18904($s3)  # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        sdc2      $20,23157($s2)  # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        sdc2      $20,-1024($s2)  # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        sdc3      $12,5835($t2)   # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
