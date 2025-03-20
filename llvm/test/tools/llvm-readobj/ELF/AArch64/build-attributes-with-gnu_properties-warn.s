# RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t.o
# RUN: ld.lld -r %t.o 2>&1 | FileCheck --check-prefix=WARN  %s

# WARN: (.ARM.attributes): object file conatains both `.note.gnu.property` and `.ARM.attributes` subsections. `.ARM.attributes` subsection ignored.

.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 49
.aeabi_attribute Tag_PAuth_Schema, 19
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 1

.section ".note.gnu.property", "a"
  .long 4           // Name length is always 4 ("GNU")
  .long end - begin // Data length
  .long 5           // Type: NT_GNU_PROPERTY_TYPE_0
  .asciz "GNU"      // Name
  .p2align 3
begin:
  # PAuth ABI property note
  .long 0xc0000001  // Type: GNU_PROPERTY_AARCH64_FEATURE_PAUTH
  .long 16          // Data size
  .quad 0           // PAuth ABI platform
  .quad 0           // PAuth ABI version
  .p2align 3        // Align to 8 byte for 64 bit
end:
.section ".note.gnu.property", "a"
.long 4
.long 0x10
.long 0x5
.asciz "GNU"
.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 4
.long 3          // GNU_PROPERTY_AARCH64_FEATURE_1_BTI and PAC
.long 0
