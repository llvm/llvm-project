// Every local below is a distinct concrete gnustep-base class, so the data
// formatters for strings in all their storage forms, boxed and tagged
// numbers, arrays, dictionaries, sets, data, dates and NSNull each get
// exercised, plus a custom object.

#import <Foundation/Foundation.h>

@interface Account : NSObject {
  NSString *owner;
  NSNumber *balance;
  NSArray *tags;
}
- (instancetype)initWithOwner:(NSString *)o balance:(double)b;
@end

@implementation Account
- (instancetype)initWithOwner:(NSString *)o balance:(double)b {
  if ((self = [super init])) {
    owner = o;
    balance = @(b);
    tags = @[ @"premium", @"verified" ];
  }
  return self;
}
- (NSString *)description {
  return [NSString stringWithFormat:@"<Account %@: %@>", owner, balance];
}
@end

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    // Strings: each literal/operation lands in a different concrete class.
    NSString *tinyString = @"Hi";                            // GSTinyString
    // A tiny string may hold any 7-bit character, including ones the
    // summary has to escape rather than emit raw.
    NSString *tinyQuoted = @"a\"b";                          // GSTinyString
    NSString *constantString = @"A constant string literal"; // NSConstantString
    NSString *unicodeConstant = @"Grüße, 世界"; // NSConstantString, UTF-16
    NSString *emptyString = @"";
    NSString *builtString =
        [NSString stringWithFormat:@"built %d", 42]; // GSCInlineString
    NSString *unicodeBuilt =
        [NSString stringWithFormat:@"ünïcödé %d", 7]; // GSUInlineString
    NSMutableString *mutableString =
        [NSMutableString stringWithString:@"mutable"]; // GSMutableString
    [mutableString appendString:@" string"];

    // Numbers: singletons, tagged small objects, and heap boxes.
    NSNumber *boolYes = @YES;                    // NSBoolNumber
    NSNumber *smallInt = @5;                     // NSIntNumber (singleton)
    NSNumber *taggedInt = @123456;               // NSSmallInt
    NSNumber *negativeInt = @-99;                // NSSmallInt
    NSNumber *longLong = @9223372036854775807LL; // NSLongLongNumber
    NSNumber *unsignedLongLong =
        @18446744073709551615ULL;      // NSUnsignedLongLongNumber
    NSNumber *floatNumber = @1.5f;     // NSSmallFloat
    NSNumber *doubleNumber = @3.14159; // NSSmallRepeatingDouble
    NSNumber *heapDouble = @0.1; // NSSmallExtendedDouble or NSDoubleNumber

    // Collections.
    NSArray *emptyArray = @[];
    NSArray *fruits = @[ @"apple", @"banana", @"cherry" ]; // GSInlineArray
    NSMutableArray *mutableArray = [NSMutableArray arrayWithArray:fruits];
    [mutableArray addObject:@"date"]; // GSMutableArray
    NSArray *nested = @[ fruits, @[ @1, @2 ] ];
    NSDictionary *emptyDict = @{};
    NSDictionary *person = @{
      @"name" : @"John Doe",
      @"age" : @30,
      @"skills" : @[ @"Objective-C", @"Swift" ]
    }; // GSDictionary
    NSMutableDictionary *mutableDict =
        [NSMutableDictionary dictionaryWithDictionary:person];
    mutableDict[@"city"] = @"Berlin"; // GSMutableDictionary
    NSSet *colors =
        [NSSet setWithObjects:@"red", @"green", @"blue", nil]; // GSSet
    NSMutableSet *mutableSet = [NSMutableSet setWithSet:colors];
    [mutableSet addObject:@"yellow"]; // GSMutableSet
    NSCountedSet *counted = [NSCountedSet setWithArray:@[ @"a", @"a", @"b" ]];

    // Other value types.
    NSData *data = [@"Hello, data!"
        dataUsingEncoding:NSUTF8StringEncoding]; // NSDataMalloc
    NSDate *epoch =
        [NSDate dateWithTimeIntervalSinceReferenceDate:0]; // GSSmallDate
    NSDate *someDate = [NSDate dateWithTimeIntervalSince1970:1700000000];
    NSNull *null = [NSNull null];
    NSURL *url = [NSURL URLWithString:@"https://www.gnustep.org/resources"];
    id nilObject = nil;

    // A custom class: gets dynamic type + ivars, po runs -description.
    Account *account = [[Account alloc] initWithOwner:@"Jane" balance:1234.5];
    id anonymous = account;

    // Message sends to step into (through objc_msgSend): one into
    // gnustep-base, one into this file.
    NSUInteger fruitCount = [fruits count];        // step here: Foundation
    NSString *accountText = [account description]; // step here: user class

    NSLog(@"%@ %@", tinyQuoted, tinyQuoted);
    NSLog(@"%@ %@ %@ %@ %@ %@ %@", tinyString, constantString, unicodeConstant,
          emptyString, builtString, unicodeBuilt, mutableString);
    NSLog(@"%@ %@ %@ %@ %@ %@ %@ %@ %@", boolYes, smallInt, taggedInt,
          negativeInt, longLong, unsignedLongLong, floatNumber, doubleNumber,
          heapDouble);
    NSLog(@"%@ %@ %@ %@ %@ %@ %@ %@ %@ %@", emptyArray, fruits, mutableArray,
          nested, emptyDict, person, mutableDict, colors, mutableSet, counted);
    NSLog(@"%@ %@ %@ %@ %@ %@ %@ %lu %@", data, epoch, someDate, null, url,
          account, anonymous, (unsigned long)fruitCount, accountText);
    return nilObject != nil; // break here
  }
}
