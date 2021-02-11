#import <Foundation/Foundation.h>

typedef struct {
    unsigned char fieldOne : 1;
    unsigned char fieldTwo : 1;
    unsigned char fieldThree : 1;
    unsigned char fieldFour : 1;
    unsigned char fieldFive : 1;
} UCBitFields;

@interface HasBitfield : NSObject {
@public
    unsigned field1 : 1;
    unsigned field2 : 1;
};

-(id)init;
@end

@implementation HasBitfield
-(id)init {
    self = [super init];
    field1 = 0;
    field2 = 1;
    return self;
}
@end

@interface ContainsAHasBitfield : NSObject {
@public
    HasBitfield *hb;
};
-(id)init;
@end

@implementation  ContainsAHasBitfield
-(id)init {
    self = [super init];
    hb = [[HasBitfield alloc] init];
    return self;
}

@end

@interface HasBitfield2 : NSObject {
@public
  unsigned int x;

  unsigned field1 : 15;
  unsigned field2 : 4;
  unsigned field3 : 4;
}
@end

@implementation HasBitfield2
- (id)init {
  return (self = [super init]);
}
@end

int main(int argc, const char * argv[]) {
    ContainsAHasBitfield *chb = [[ContainsAHasBitfield alloc] init];
    HasBitfield2 *hb2 = [[HasBitfield2 alloc] init];

    hb2->x = 100;
    hb2->field1 = 10;
    hb2->field2 = 3;
    hb2->field3 = 4;

    UCBitFields myField = {0};
    myField.fieldTwo = 1;
    myField.fieldFive = 1;

    return 0; // break here
}

