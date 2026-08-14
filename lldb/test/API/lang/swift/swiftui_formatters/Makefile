SWIFT_SOURCES := main.swift

# On the macOS 26+ SDK, SwiftUI's @State is an external macro
# (SwiftUIMacros.StateMacro). Expanding it launches swift-plugin-server, which
# swift wraps in `sandbox-exec`; sandbox_apply is rejected on recent macOS, so
# the plugin server can't start ("malformed response") and the macro fails to
# resolve. Disable the macro-plugin sandbox so the test subject builds.
SWIFTFLAGS_EXTRAS := -Xfrontend -disable-sandbox
SWIFT_DEPLOYMENT_TARGET := 26

include Makefile.rules
