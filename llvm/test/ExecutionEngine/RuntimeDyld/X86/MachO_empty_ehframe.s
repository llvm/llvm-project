# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %T/empty_eh_frame.o %s
# RUN: llvm-rtdyld -verify -triple=x86_64-apple-macosx10.9 %T/empty_eh_frame.o

        .section        __TEXT,__eh_frame
	.macosx_version_min 10, 10

.subsections_via_symbols
