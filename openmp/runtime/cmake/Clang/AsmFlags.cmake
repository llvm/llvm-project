# This file holds Clang (clang/clang++) specific compiler dependent flags
# The flag types are:
#   1) Assembly flags

#########################################################
# Assembly flags
function(append_assembler_specific_asm_flags input_asm_flags)
    set(local_asm_flags)
    append_asm_flags("-x assembler-with-cpp") # Assembly file that needs to be preprocessed
    if(${IA32})
        append_asm_flags("-m32") # Generate 32 bit IA-32 architecture code 
        append_asm_flags("-msse2") # Allow use of Streaming SIMD Instructions
    endif()
    set(${input_asm_flags} ${${input_asm_flags}} "${local_asm_flags}" PARENT_SCOPE)
endfunction()

