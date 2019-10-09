#import <Foundation/Foundation.h>

enum CMYK { cyan, magenta, yellow, black };

typedef enum CMYK FourColors;

union Union { int i; };

@interface ObjCClass : NSNumber
- (NSString * _Nonnull)debugDescription;
@end

typedef NSString * _Nonnull OBJCSTUFF_MyString
__attribute((swift_newtype(struct))) __attribute((swift_name("MyString")));

typedef float MyFloat __attribute((swift_newtype(struct)));
extern const MyFloat globalFloat;
