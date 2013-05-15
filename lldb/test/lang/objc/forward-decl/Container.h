#import <Foundation/Foundation.h>

@class ForwardDeclaredClass;

@interface Container : NSObject {
@public
    ForwardDeclaredClass *member;
}

-(id)init;
-(ForwardDeclaredClass*)getMember;

@end
