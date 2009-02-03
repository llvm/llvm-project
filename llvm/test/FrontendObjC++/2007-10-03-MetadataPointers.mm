// RUN: %llvmgcc -w -x objective-c++ -c %s -o /dev/null

@class NSImage;
void bork() {
  NSImage *nsimage;
  [nsimage release];
}
