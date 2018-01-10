# RUN: not llvm-mc -triple x86_64-unknown-unknown -dwarf-version 5 -filetype=asm %s -o /dev/null 2>&1 | FileCheck %s

# This is syntactically legal, looks like no checksum provided.
# CHECK-NOT: [[@LINE+1]]:{{[0-9]+}}: error:
        .file 1 "dir1/foo" "00112233445566778899aabbccddeeff"

# Missing md5 keyword.
# CHECK: [[@LINE+1]]:{{[0-9]+}}: error: unexpected token in '.file' directive
        .file 2 "dir1" "foo" "00112233445566778899aabbccddeeff"

# Bad length.
# CHECK: [[@LINE+1]]:{{[0-9]+}}: error: invalid MD5 checksum specified
        .file 3 "dir2" "bar" md5 "ff"

# Not a string.
# CHECK: [[@LINE+1]]:{{[0-9]+}}: error: unexpected token in '.file' directive
        .file 4 "dir3" "foo" md5 ffeeddccbbaa99887766554433221100

# Non-DWARF .file syntax with checksum.
# CHECK: [[@LINE+1]]:{{[0-9]+}}: error: MD5 checksum specified, but no file number
        .file "baz" md5 "ffeeddccbbaa998877665544332211gg"
