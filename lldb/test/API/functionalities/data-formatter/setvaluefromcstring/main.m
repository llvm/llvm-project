#import <Foundation/Foundation.h>

int main() {
    NSDictionary* dic = @{@1 : @2};
    BOOL b = NO;
    NSLog(@"hello world"); //% dic = self.frame().FindVariable("dic")
    //% dic.SetPreferSyntheticValue(True)
    //% dic.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
    //% dic.SetValueFromCString("12")
    //% b = self.frame().FindVariable("b")
    //% b.SetValueFromCString("YES")
    return 0; //% dic = self.frame().FindVariable("dic")
    //% self.assertTrue(dic.GetValueAsUnsigned() == 0xC, "failed to read what I wrote")
    //% b = self.frame().FindVariable("b")
    //% self.assertTrue(b.GetValueAsUnsigned() == 0x0, "failed to update b")
}
