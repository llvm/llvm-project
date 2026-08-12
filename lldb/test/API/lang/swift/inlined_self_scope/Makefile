SWIFT_SOURCES := main.swift

# -O is required: the bug only manifests when the PC is inside an inlined
# function. The test asserts that the frame really is inlined, so it fails
# loudly rather than silently passing if the optimizer stops inlining here.
SWIFTFLAGS_EXTRAS := -O

include Makefile.rules
