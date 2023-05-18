#import <Foundation/Foundation.h>

int main() {
  NSLog(@"凸"); //% self.expect("po @\"凹\"", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["凹"])
}
