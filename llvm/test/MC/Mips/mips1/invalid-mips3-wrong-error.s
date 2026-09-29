# Instructions that are invalid and are correctly rejected but used to emit
# the wrong error message.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips1 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        ldc2      $8,-21181($at)    # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        ldc2      $20,-1024($s2)    # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        ldl       $24,-4167($24)    # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        ldr       $14,-30358($s4)   # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        ll        $v0,-7321($s2)    # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        sc        $15,18904($s3)    # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        scd       $15,-8243($sp)    # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        sdc2      $20,23157($s2)    # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        sdc2      $20,-1024($s2)    # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        sdl       $a3,-20961($s8)   # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
        sdr       $11,-20423($12)   # CHECK: :[[#@LINE]]:[[#]]: error: instruction requires a CPU feature not currently enabled 
