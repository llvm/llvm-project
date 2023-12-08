/*
clang -g ExtendSuperclass.m -o ExtendSuperclass -framework Foundation -framework ProtectedCloudStorage -F/System/Library/PrivateFrameworks/ -framework CloudKit && ./ExtendSuperclass
*/
#include <assert.h>
#import <Foundation/Foundation.h>
#import <CloudKit/CloudKit.h>

#define SuperClass CKDatabase

@interface SubClass : SuperClass
@end

// class extension
@interface SuperClass ()
@property (nonatomic, strong)     NSString             *_sc_name;
@property (nonatomic, strong)     NSString             *_sc_name2;
@property (nonatomic, strong)     NSString             *_sc_name3;
@property (nonatomic, strong)     NSString             *_sc_name4;
@property (nonatomic, strong)     NSString             *_sc_name5;
@property (nonatomic, strong)     NSString             *_sc_name6;
@property (nonatomic, strong)     NSString             *_sc_name7;
@property (nonatomic, strong)     NSString             *_sc_name8;
@end

@implementation SuperClass (MySuperClass)
- (id)initThatDoesNotAssert
{
    return [super init];
}
@end

@implementation SubClass
- (id)initThatDoesNotAssert
{
    assert(_sc_name == nil);
    assert(_sc_name2 == nil);
    assert(_sc_name3 == nil);
    assert(_sc_name4 == nil);
    assert(_sc_name5 == nil);
    assert(_sc_name6 == nil);
    assert(_sc_name7 == nil);
    assert(_sc_name8 == nil); // break here

    if ((self = [super _initWithContainer:(CKContainer*)@"foo" scope:0xff])) {
        assert(_sc_name == nil);
        assert(_sc_name2 == nil);
        assert(_sc_name3 == nil);
        assert(_sc_name4 == nil);
        assert(_sc_name5 == nil);
        assert(_sc_name6 == nil);
        assert(_sc_name7 == nil);
        assert(_sc_name8 == nil);

        _sc_name = @"empty";
    }
    return self;
}
@synthesize _sc_name;
@synthesize _sc_name2;
@synthesize _sc_name3;
@synthesize _sc_name4;
@synthesize _sc_name5;
@synthesize _sc_name6;
@synthesize _sc_name7;
@synthesize _sc_name8;
- (void)foo:(NSString*)bar { self._sc_name = bar; }
- (NSString*)description { return [NSString stringWithFormat:@"%p: %@", self, self._sc_name]; }
@end

int main()
{
    SubClass *sc = [[SubClass alloc] initThatDoesNotAssert];
    NSLog(@"%@", sc);
    [sc foo:@"bar"];
    NSLog(@"%@", sc);
    sc._sc_name = @"bar2";
    NSLog(@"%@", sc);
    NSLog(@"%@", sc._sc_name);
    return 0;
}
