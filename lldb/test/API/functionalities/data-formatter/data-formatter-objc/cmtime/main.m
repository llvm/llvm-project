#import <CoreMedia/CoreMedia.h>

int main(int argc, const char **argv)
{
    @autoreleasepool
    {
        CMTime t1 = CMTimeMake(1, 2);
        CMTime t2 = CMTimeMake(1, 3);
        CMTime t3 = CMTimeMake(1, 10);
        CMTime t4 = CMTimeMake(10, 1);
        CMTime t5 = CMTimeMake(10, 1);
        t5.flags = kCMTimeFlags_PositiveInfinity;
        CMTime t6 = CMTimeMake(10, 1);
        t6.flags = kCMTimeFlags_NegativeInfinity;
        CMTime t7 = CMTimeMake(10, 1);
        t7.flags = kCMTimeFlags_Indefinite;

        CMTimeShow(t1); // break here
        CMTimeShow(t2);
        CMTimeShow(t3);
        CMTimeShow(t4);
        CMTimeShow(t5);
        CMTimeShow(t6);
        CMTimeShow(t7);
    }
    return 0;
}
