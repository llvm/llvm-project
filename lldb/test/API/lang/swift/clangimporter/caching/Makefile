SWIFT_SOURCES := main.swift
SWIFT_ENABLE_EXPLICIT_MODULES := YES
SWIFTFLAGS_EXTRAS = -I$(SRCDIR) -I/TEST_DIR -F/FRAMEWORK_DIR -cache-compile-job -cas-path $(BUILDDIR)/cas

all: a.out cas-config

include Makefile.rules

CAS_CONFIG_CONTENT := {"CASPath": "$(BUILDDIR)/cas"}

cas-config:
	$(call ECHO_TO_FILE,'$(CAS_CONFIG_CONTENT)',.cas-config)
