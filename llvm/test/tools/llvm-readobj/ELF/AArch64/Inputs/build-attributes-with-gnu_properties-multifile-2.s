
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
  .quad 49           // PAuth ABI platform
  .quad 19           // PAuth ABI version
  .p2align 3        // Align to 8 byte for 64 bit
end:
