SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -parse-stdlib -Xcc -DMARKER1 -Xcc -fno-implicit-modules -Xcc -fno-implicit-module-maps -Xcc -DMARKER2

# -parse-stdlib tells the Swift driver not to autolink the runtime, but the
# emitted object still references swift_addNewDSOImage.
# On Windows lld-link leaves that dllimport unresolved, so link swiftCore
# explicitly.
ifeq "$(OS)" "Windows_NT"
LD_EXTRAS = -lswiftCore
endif

include Makefile.rules
