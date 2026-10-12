#!/bin/sh
# Dummy dsymForUUID for the LLDB test suite.
# Returns an empty plist to indicate that no dSYM was found, without making
# any network requests or slow filesystem searches.
echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
echo "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">"
echo "<plist version=\"1.0\">"
echo "<dict/>"
echo "</plist>"
