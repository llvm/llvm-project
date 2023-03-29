#include "stdio.h"

#define TYPE double
#define PRINT_PRECISION_FORMAT "%0.15lf"

TYPE compute_barycentric_coordinates(TYPE ax, TYPE ay, TYPE az,
                                       TYPE bx, TYPE by, TYPE bz,
                                       TYPE cx, TYPE cy, TYPE cz,
                                       TYPE px, TYPE py, TYPE pz) {
  TYPE bax = bx - ax ;
  TYPE bay = by - ay ;
  TYPE baz = bz - az ;
  TYPE cax = cx - ax ;
  TYPE cay = cy - ay ;
  TYPE caz = cz - az ;
  TYPE mx  = bay*caz - baz*cay;
  TYPE my = baz*cax - bax*caz;
  TYPE mz = bax*cay - bay*cax;

  TYPE nu , nv, ood, u, v, w ;

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

  return w;
}

int main(void)
{

  TYPE ax;
  TYPE ay;
  TYPE az;
  TYPE bx;
  TYPE by;
  TYPE bz;
  TYPE cx;
  TYPE cy;
  TYPE cz;
  TYPE px;
  TYPE py;
  TYPE pz;

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

  TYPE w = compute_barycentric_coordinates(ax, ay, az, bx, by, bz, cx, cy, cz, px, py, pz);

  printf("Result = "PRINT_PRECISION_FORMAT"\n", w);

  return 0;

}
