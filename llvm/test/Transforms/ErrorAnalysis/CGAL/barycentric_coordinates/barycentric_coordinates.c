#include "stdio.h"

int main(void)
{

  double ax;
  double ay;
  double az;
  double bx;
  double by;
  double bz;
  double cx;
  double cy;
  double cz;
  double px;
  double py;
  double pz;

  ax = -4.157;
  ay = 4.93;
  az = -6.73;
  bx = 7.393;
  by = -7.17;
  bz = 8.42;
  cx = 3.941;
  cy = 7.84;
  cz = -3.85;
  px = 1.5;
  py = 2.5;
  pz = 4.5;

  double bax = bx - ax ;
  double bay = by - ay ;
  double baz = bz - az ;
  double cax = cx - ax ;
  double cay = cy - ay ;
  double caz = cz - az ;
  double mx  = bay*caz - baz*cay;
  double my = baz*cax - bax*caz;
  double mz = bax*cay - bay*cax;

  double nu , nv, ood, u, v, w ;

  if(( ((mx >= 0) && (my >= 0) && (mx >= my)) || ((mx < 0) && (my >= 0) && (mx+my <= 0)) ||
       ((mx >= 0) && (my < 0) && (mx + my >= 0)) || ((mx < 0) && (my < 0) && (mx - my <= 0))
           ) && ( ((mx >= 0) && (mz >= 0) && (mx >= mz)) || ((mx < 0) && (mz >= 0) && (mx+mz <= 0)) ||
       ((mx >= 0) && (mz < 0) && (mx + mz >= 0)) || ((mx < 0) && (mz < 0) && (mx - mz <= 0))
           )) {
    nu = (py-by)*(bz-cz) - (by-cy)*(pz-bz);
    nv = (py-cy)*(cz-az) - (cy-ay)*(pz-cz);
    ood = 1.0/mx ;
  }	else {

    if(( ((my >= 0) && (mx >= 0) && (my >= mx)) || ((my < 0) && (mx >= 0) && (my+mx <= 0)) ||
         ((my >= 0) && (mx < 0) && (my + mx >= 0)) || ((my < 0) && (mx < 0) && (my - mx <= 0))
             ) && ( ((my >= 0) && (mz >= 0) && (my >= mz)) || ((my < 0) && (mz >= 0) && (my+mz <= 0)) ||
         ((my >= 0) && (mz < 0) && (my + mz >= 0)) || ((my < 0) && (mz < 0) && (my - mz <= 0))
             )) {
      nu = (px - bx)*(bz - cz) - (bx - cx)*(pz-bz);
      nv = (px - cx)*(cz - az) - (cx-ax)*(pz-cz);
      ood = 1.0/(-1.0*my);
    } else {
      nu = (px-bx)*(by-cy) - (bx-cx)*(py-by);
      nv = (px - cx)*(cy-ay) - (cx - ax)*(py - cy);
      ood = 1.0/mz ;
    }

  }

  u = nu * ood ;
  v = nv * ood ;
  w = 1.0 - u - v ;

  printf("Result = %0.15lf\n", w);

  return 0;

}
